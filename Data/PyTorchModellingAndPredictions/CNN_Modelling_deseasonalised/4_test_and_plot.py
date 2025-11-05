import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

ROOT = "region_splits"
OUT_CSV_NAME = "test_temp_pred_actual.csv"
OUT_PNG_NAME = "temp_vs_actual_and_pred_TEST.png"
OUT_PNG_PRED_ONLY_NAME = "temp_vs_pred_ONLY_TEST.png"

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

#use best device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch, "mps", None) and torch.mps.is_available():
    device = torch.device("mps")

#training NN model
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

regions = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

for reg in regions:
	reg_dir = os.path.join(ROOT, reg)
	test_path = os.path.join(reg_dir, "test.csv")
	model_path = os.path.join(reg_dir, "model.pt")
	prep_path  = os.path.join(reg_dir, "preprocess.npz")

	#Load test data
	df = pd.read_csv(test_path)

	#Load preprocess (training feature order + stats)
	pp = np.load(prep_path, allow_pickle=True)
	feat_cols = [str(x) for x in pp["feat_cols"].tolist()]  # use training feature order
	mu = pd.Series(pp["mu"], index=feat_cols)
	sd = pd.Series(pp["sd"], index=feat_cols).replace(0, 1.0)

	#target standardization
	y_mu = float(pp["y_mu"][0]) if "y_mu" in pp.files else None
	y_sd = float(pp["y_sd"][0]) if "y_sd" in pp.files else None

	#Build input matrix in training order
	cols = {}
	for c in feat_cols:
		if c in df.columns:
			cols[c] = pd.to_numeric(df[c], errors="coerce")
		else:
			cols[c] = pd.Series(mu[c], index=df.index)
	X_df = pd.DataFrame(cols)
	if "snowdepth" in X_df.columns:
		X_df["snowdepth"] = X_df["snowdepth"].fillna(0.0)

	#target + temp for plotting
	y = pd.to_numeric(df.get("excess_deaths", pd.Series(index=df.index, dtype=float)), errors="coerce")
	t = pd.to_numeric(df.get("tempmax", pd.Series(index=df.index, dtype=float)), errors="coerce")

	mask = X_df.notna().all(axis=1) & y.notna() & t.notna()
	X_df = X_df[mask].reset_index(drop=True)
	y = y[mask].reset_index(drop=True)
	t = t[mask].reset_index(drop=True)

	#standardize using training stats
	Xn = (X_df - mu) / sd
	X = torch.tensor(Xn.to_numpy(float), dtype=torch.float32, device=device)

	#load model and predict
	model = MLP(in_feats=len(feat_cols), dropout_p=0.2).to(device)
	state = torch.load(model_path, map_location=device)
	model.load_state_dict(state, strict=True)
	model.eval()
	with torch.no_grad():
		yhat = model(X).cpu().view(-1).numpy()

	#de-standardize
	yhat = yhat * y_sd + y_mu

	#Save CSV
	out_df = pd.DataFrame({"tempmax": t.to_numpy(float),"predicted_excess": yhat,"actual_excess": y.to_numpy(float),})
	out_csv = os.path.join(reg_dir, OUT_CSV_NAME)
	out_df.to_csv(out_csv, index=False)

	#Plot pred + actual + histogram
	fig, (ax_scatter, ax_bar) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0})
	ax_scatter.scatter(out_df["tempmax"], out_df["predicted_excess"],s=4, alpha=1., label="Predicted (TEST)", zorder = 1)
	ax_scatter.scatter(out_df["tempmax"], out_df["actual_excess"],s=4, alpha=1., label="Actual (TEST)", zorder = 0)
	ax_scatter.set_ylabel("Excess deaths", labelpad=-1)
	ax_scatter.set_title(f"{reg} — Tempmax vs Excess Deaths")
	ax_scatter.legend()

	counts, bins = np.histogram(out_df["tempmax"], bins=30)
	ax_bar.bar(bins[:-1], counts, width=np.diff(bins), align='edge')
	ax_bar.set_ylabel("Count")
	ax_bar.set_xlabel("Mean daily maximum temperature (°C)")

	fig.savefig(os.path.join(reg_dir, OUT_PNG_NAME), dpi=350)
	plt.close(fig)

	#Plot predictions only + histogram
	fig2, (ax_scatter2, ax_bar2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0})
	ax_scatter2.scatter(out_df["tempmax"], out_df["predicted_excess"],s=4, alpha=1.)
	ax_scatter2.set_ylabel("Predicted excess deaths", labelpad=-1)
	ax_scatter2.set_title(f"{reg} — Tempmax vs Predicted Excess")

	counts2, bins2 = np.histogram(out_df["tempmax"], bins=30)
	ax_bar2.bar(bins2[:-1], counts2, width=np.diff(bins2), align='edge')
	ax_bar2.set_ylabel("Count")
	ax_bar2.set_xlabel("Mean daily maximum temperature (°C)")

	fig2.savefig(os.path.join(reg_dir, OUT_PNG_PRED_ONLY_NAME), dpi=350)
	plt.close(fig2)