import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
plt.rcParams.update({'font.size': 12})

ROOT = "region_splits"
OUT_CSV_NAME = "test_temp_pred_actual.csv"
OUT_SUMMARY_CSV = "PyTorch_model_fit_metrics_summary.csv"

list_of_regions = [
    "NorthEast","NorthWest","YorkshireandtheHumber","EastMidlands","WestMidlands",
    "East","London","SouthEast","SouthWest","Wales"
]

list_of_region_names_for_plots = [
    "North East","North West","Yorkshire and the Humber","East Midlands","West Midlands",
    "East","London","South East","South West","Wales"
]

list_of_region_codes = [
    "E12000001","E12000002","E12000003","E12000004","E12000005",
    "E12000006","E12000007","E12000008","E12000009","W92000004"
]

#not in sklearn:
def pseudo_r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    y0 = y - np.mean(y); h0 = yhat - np.mean(yhat)
    den = np.sqrt(np.sum(y0**2) * np.sum(h0**2))
    return (np.sum(y0*h0)/den)**2

def zero_baseline(y):
    y = np.asarray(y, float)
    mse0 = np.mean(y**2)
    rmse0 = np.sqrt(mse0)
    return mse0, rmse0

regions = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
print("Working dir:", os.getcwd())
print("Regions found:", regions)

rows = []

#accumulate for TOTAL
Ytr_all = []
Ytrhat_all = []
Yte_all = []
Ytehat_all = []
RESID_all = []

for reg in regions:
    reg_dir = os.path.join(ROOT, reg)
    train_csv = os.path.join(reg_dir, "train.csv")
    test_csv  = os.path.join(reg_dir, "test.csv")
    pred_csv  = os.path.join(reg_dir, OUT_CSV_NAME)
    prep_path = os.path.join(reg_dir, "preprocess.npz")
    model_path = os.path.join(reg_dir, "model.pt")

    i = list_of_region_codes.index(reg)
    reg_name = list_of_region_names_for_plots[i]

    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)
    df_pred  = pd.read_csv(pred_csv)
    if "pred_minus_actual" not in df_pred.columns:
        df_pred["pred_minus_actual"] = df_pred["predicted_excess"] - df_pred["actual_excess"]

    y_train_true = pd.to_numeric(df_train["excess_deaths"], errors="coerce").to_numpy(float)
    y_test_true  = pd.to_numeric(df_pred["actual_excess"], errors="coerce").to_numpy(float)
    y_test_hat   = pd.to_numeric(df_pred["predicted_excess"], errors="coerce").to_numpy(float)
    y_resid_vec  = pd.to_numeric(df_pred["pred_minus_actual"], errors="coerce").to_numpy(float)

    pp = np.load(prep_path, allow_pickle=True)
    feat_cols = [str(x) for x in pp["feat_cols"].tolist()]
    mu = pd.Series(pp["mu"], index=feat_cols)
    sd = pd.Series(pp["sd"], index=feat_cols).replace(0, 1.0)
    y_mu = float(pp["y_mu"][0])
    y_sd = float(pp["y_sd"][0])

    cols_tr = {}
    for c in feat_cols:
        cols_tr[c] = pd.to_numeric(df_train[c], errors="coerce") if c in df_train.columns else pd.Series(mu[c], index=df_train.index)
    Xtr_df = pd.DataFrame(cols_tr)
    if "snowdepth" in Xtr_df.columns:
        Xtr_df["snowdepth"] = Xtr_df["snowdepth"].fillna(0.0)
    Xtrn = (Xtr_df - mu) / sd

    import torch
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, in_feats:int, dropout_p:float=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_feats, 64), nn.ReLU(), nn.Dropout(dropout_p),
                nn.Linear(64, 32),      nn.ReLU(), nn.Dropout(dropout_p),
                nn.Linear(32, 1),
            )
        def forward(self, x): return self.net(x)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch, "mps", None) and torch.mps.is_available():
        device = torch.device("mps")

    model = MLP(in_feats=len(feat_cols), dropout_p=0.2).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        Xtr_t = torch.tensor(Xtrn.to_numpy(float), dtype=torch.float32, device=device)
        y_train_hat_z = model(Xtr_t).cpu().view(-1).numpy()
    y_train_hat = y_train_hat_z * y_sd + y_mu

    n_train = int(len(y_train_true))
    n_test  = int(len(y_test_true))

    mse_train = mean_squared_error(y_train_true, y_train_hat)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train_true, y_train_hat)
    nmse_train = mse_train / np.var(y_train_true)
    r2_train = r2_score(y_train_true, y_train_hat)
    pr2_train = pseudo_r2(y_train_true, y_train_hat)
    mse0_tr, rmse0_tr = zero_baseline(y_train_true)
    improve0_tr = 1.0 - mse_train / mse0_tr
    better_than_zero_tr = (mse_train < mse0_tr)

    mse_test = mean_squared_error(y_test_true, y_test_hat)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test_true, y_test_hat)
    nmse_test = mse_test / np.var(y_test_true)
    r2_test = r2_score(y_test_true, y_test_hat)
    pr2_test = pseudo_r2(y_test_true, y_test_hat)
    mse0_te, rmse0_te = zero_baseline(y_test_true)
    improve0_te = 1.0 - mse_test / mse0_te
    better_than_zero_te = (mse_test < mse0_te)

    resid = y_resid_vec
    residual_bias = float(np.mean(resid))
    residual_std  = float(np.std(resid, ddof=0))

    rows.append({
        "region_code": reg,
        "region_name": reg_name,
        "n_train": n_train, "n_test": n_test,
        "r2_train": r2_train, "r2_test": r2_test,
        "pseudo_r2_train": pr2_train, "pseudo_r2_test": pr2_test,
        "rmse_train": rmse_train, "rmse_test": rmse_test,
        "mae_train": mae_train,   "mae_test": mae_test,
        "nmse_train": nmse_train, "nmse_test": nmse_test,
        "zero_rmse_train": rmse0_tr, "zero_rmse_test": rmse0_te,
        "zero_mse_train": mse0_tr,   "zero_mse_test": mse0_te,
        "improve_vs_zero_train": improve0_tr, "improve_vs_zero_test": improve0_te,
        "better_than_zero_train": better_than_zero_tr, "better_than_zero_test": better_than_zero_te,
        "residual_bias": residual_bias, "residual_std": residual_std,
    })

    # accumulate for TOTAL
    Ytr_all.append(y_train_true)
    Ytrhat_all.append(y_train_hat)
    Yte_all.append(y_test_true)
    Ytehat_all.append(y_test_hat)
    RESID_all.append(resid)

summary = pd.DataFrame(rows)

# --- TOTAL row (compute metrics on concatenated arrays) ---
Ytr_all = np.concatenate(Ytr_all) if len(Ytr_all) else np.array([])
Ytrhat_all = np.concatenate(Ytrhat_all) if len(Ytrhat_all) else np.array([])
Yte_all = np.concatenate(Yte_all) if len(Yte_all) else np.array([])
Ytehat_all = np.concatenate(Ytehat_all) if len(Ytehat_all) else np.array([])
RESID_all = np.concatenate(RESID_all) if len(RESID_all) else np.array([])

n_train_T = int(len(Ytr_all))
n_test_T  = int(len(Yte_all))

mse_train_T = mean_squared_error(Ytr_all, Ytrhat_all)
rmse_train_T = np.sqrt(mse_train_T)
mae_train_T = mean_absolute_error(Ytr_all, Ytrhat_all)
nmse_train_T = mse_train_T / np.var(Ytr_all)
r2_train_T = r2_score(Ytr_all, Ytrhat_all)
pr2_train_T = pseudo_r2(Ytr_all, Ytrhat_all)
mse0_tr_T, rmse0_tr_T = zero_baseline(Ytr_all)
improve0_tr_T = 1.0 - mse_train_T / mse0_tr_T
better_than_zero_tr_T = (mse_train_T < mse0_tr_T)

mse_test_T = mean_squared_error(Yte_all, Ytehat_all)
rmse_test_T = np.sqrt(mse_test_T)
mae_test_T = mean_absolute_error(Yte_all, Ytehat_all)
nmse_test_T = mse_test_T / np.var(Yte_all)
r2_test_T = r2_score(Yte_all, Ytehat_all)
pr2_test_T = pseudo_r2(Yte_all, Ytehat_all)
mse0_te_T, rmse0_te_T = zero_baseline(Yte_all)
improve0_te_T = 1.0 - mse_test_T / mse0_te_T
better_than_zero_te_T = (mse_test_T < mse0_te_T)

residual_bias_T = float(np.mean(RESID_all))
residual_std_T  = float(np.std(RESID_all, ddof=0))

total_row = {
    "region_code": "TOTAL",
    "region_name": "Total",
    "n_train": n_train_T, "n_test": n_test_T,
    "r2_train": r2_train_T, "r2_test": r2_test_T,
    "pseudo_r2_train": pr2_train_T, "pseudo_r2_test": pr2_test_T,
    "rmse_train": rmse_train_T, "rmse_test": rmse_test_T,
    "mae_train": mae_train_T,   "mae_test": mae_test_T,
    "nmse_train": nmse_train_T, "nmse_test": nmse_test_T,
    "zero_rmse_train": rmse0_tr_T, "zero_rmse_test": rmse0_te_T,
    "zero_mse_train": mse0_tr_T,   "zero_mse_test": mse0_te_T,
    "improve_vs_zero_train": improve0_tr_T, "improve_vs_zero_test": improve0_te_T,
    "better_than_zero_train": better_than_zero_tr_T, "better_than_zero_test": better_than_zero_te_T,
    "residual_bias": residual_bias_T, "residual_std": residual_std_T,
}

summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

metric_cols = [c for c in summary.columns if c not in ["region_code","region_name","n_train","n_test","better_than_zero_train","better_than_zero_test"]]
for c in metric_cols:
    summary[c] = pd.to_numeric(summary[c], errors="coerce").round(3)

save_path = os.path.abspath(OUT_SUMMARY_CSV)
summary.to_csv(save_path, index=False)