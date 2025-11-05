import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

ROOT = "region_splits"
OUT_PNG_NAME = "expected_excess_vs_temp_shift_Jun_Dec.png"

list_of_regions = [
    "NorthEast",
    "NorthWest",
    "YorkshireandtheHumber",
    "EastMidlands",
    "WestMidlands",
    "East",
    "London",
    "SouthEast",
    "SouthWest",
    "Wales"
]

list_of_region_names_for_plots = [
    "North East",
    "North West",
    "Yorkshire and the Humber",
    "East Midlands",
    "West Midlands",
    "East",
    "London",
    "South East",
    "South West",
    "Wales"
]

list_of_region_codes = [
    "E12000001",
    "E12000002",
    "E12000003",
    "E12000004",
    "E12000005",
    "E12000006",
    "E12000007",
    "E12000008",
    "E12000009",
    "W92000004"
]

list_of_cities = [
    "newcastle-upon-tyne",
    "manchester",
    "leeds",
    "nottingham",
    "birmingham",
    "norwich",
    "london",
    "brighton-and-hove",
    "bristol",
    "cardiff"
]

#use best device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch, "mps", None) and torch.mps.is_available():
    device = torch.device("mps")

#same NN shape as training
class MLP(nn.Module):
    def __init__(self, in_feats:int, dropout_p:float=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

# temp shifts to examine (degrees C)
TEMP_SHIFTS = np.arange(-5.0, 5.0 + 0.5, 0.5)

# features will be read from each region's preprocess.npz
regions = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

for reg in regions:
    #folder name -> city name for weather files
    idx = list_of_region_codes.index(reg)
    region_key = list_of_regions[idx]
    city = list_of_cities[idx]

    reg_dir = os.path.join(ROOT, reg)
    model_path = os.path.join(reg_dir, "model.pt")
    prep_path  = os.path.join(reg_dir, "preprocess.npz")

    #load preprocess
    pp = np.load(prep_path, allow_pickle=True)
    feat_cols = [str(x) for x in pp["feat_cols"].tolist()]
    mu = pd.Series(pp["mu"], index=feat_cols)
    sd = pd.Series(pp["sd"], index=feat_cols).replace(0, 1.0)
    y_mu = float(pp["y_mu"][0]) if "y_mu" in pp.files else None
    y_sd = float(pp["y_sd"][0]) if "y_sd" in pp.files else None

    #load model
    model = MLP(in_feats=len(feat_cols), dropout_p=0.2).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    #load weather for this region (1981-2019)
    weather_data_loc = f"../../Weather_data/{region_key}/"
    weather_data_file = f"{city}_1981-01-01_to_2019-12-31.csv"
    w = pd.read_csv(weather_data_loc + weather_data_file)
    w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce")
    w["month"] = w["datetime"].dt.month

    #columns we might need
    requested = ["tempmax","tempmin","windspeed","precip","snowdepth","humidity"]
    for c in requested:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")

    #monthly medians over all years for June and December
    jun = w[w["month"] == 6][requested].median(numeric_only=True)
    dec = w[w["month"] == 12][requested].median(numeric_only=True)

    #fill any missing with 0 for snowdepth and with overall monthly median or 0 for others
    if "snowdepth" in jun.index:
        jun["snowdepth"] = 0.0 if np.isnan(jun["snowdepth"]) else jun["snowdepth"]
    if "snowdepth" in dec.index:
        dec["snowdepth"] = 0.0 if np.isnan(dec["snowdepth"]) else dec["snowdepth"]

    jun = jun.fillna(0.0)
    dec = dec.fillna(0.0)

    #build base feature vectors in training order
    def build_base(month_vec):
        cols = {}
        for c in feat_cols:
            if c in month_vec.index:
                cols[c] = month_vec[c]
            else:
                cols[c] = mu[c] if c in mu.index else 0.0
        x = pd.Series(cols, index=feat_cols, dtype=float)
        return x

    x_jun_base = build_base(jun)
    x_dec_base = build_base(dec)

    #predict from a single feature vector
    def predict_from_vec(vec_series: pd.Series):
        Xn = (vec_series - mu) / sd
        X = torch.tensor(Xn.to_numpy(float)[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            yhat_z = model(X).cpu().view(-1).numpy()[0]
        yhat = yhat_z * y_sd + y_mu
        return float(yhat)

    #sweep temp shifts for June and December (only shifting tempmax)
    #convert weekly prediction -> monthly total by multiplying by days_in_month / 7
    DAYS_JUNE = 30.0
    DAYS_DEC  = 31.0

    y_jun = []
    y_dec = []
    for dT in TEMP_SHIFTS:
        vj = x_jun_base.copy()
        if "tempmax" in vj.index:
            vj["tempmax"] = float(vj["tempmax"]) + float(dT)
        weekly_jun = predict_from_vec(vj)
        y_jun.append(weekly_jun * (DAYS_JUNE / 7.0))

        vd = x_dec_base.copy()
        if "tempmax" in vd.index:
            vd["tempmax"] = float(vd["tempmax"]) + float(dT)
        weekly_dec = predict_from_vec(vd)
        y_dec.append(weekly_dec * (DAYS_DEC / 7.0))

    #subtract the value at x=0 so curves are centered at 0
    i0 = list(TEMP_SHIFTS).index(0.0)
    y0_jun = y_jun[i0]
    y0_dec = y_dec[i0]
    y_jun = [v - y0_jun for v in y_jun]
    y_dec = [v - y0_dec for v in y_dec]

    #plot: predicted monthly total (centered) vs tempmax shift for June and December
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(TEMP_SHIFTS, y_jun, marker="o", linewidth=1.0, label="June (total)")
    ax.plot(TEMP_SHIFTS, y_dec, marker="o", linewidth=1.0, label="December (total)")
    ax.set_xlabel("Tempmax shift (°C)")
    ax.set_ylabel("Month's predicted deaths")
    ax.set_title(f"{reg} — Expected deaths vs temp shift")
    ax.legend()
    plt.tight_layout()

    out_png = os.path.join(reg_dir, OUT_PNG_NAME)
    fig.savefig(out_png, dpi=350)
    plt.close(fig)