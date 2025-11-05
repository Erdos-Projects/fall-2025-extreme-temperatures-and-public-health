# 9_simple_polyfit_rmse.py  (updated to also report MAE)
import os
import numpy as np
import pandas as pd

ROOT = "region_splits"
OUT_SUMMARY_CSV = "simple_polynimial_fit_metrics_summary.csv"  # keeps same name; now includes MAE too
DEGREE = 2  # set to 1 for straight-line fit, 2 for quadratic, etc.

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

def load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    x = pd.to_numeric(df["tempmax"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["excess_deaths"], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

regions = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

rows = []
Ytr_all = []
Ytrhat_all = []
Yte_all = []
Ytehat_all = []

for reg in regions:
    reg_dir = os.path.join(ROOT, reg)
    train_csv = os.path.join(reg_dir, "train.csv")
    test_csv  = os.path.join(reg_dir, "test.csv")

    idx = list_of_region_codes.index(reg)
    reg_name = list_of_region_names_for_plots[idx]

    #load data
    xtr, ytr = load_xy(train_csv)
    xte, yte = load_xy(test_csv)

    #fit polynomial on TRAIN only
    coef = np.polyfit(xtr, ytr, DEGREE)

    #predict on train & test
    ytr_hat = np.polyval(coef, xtr)
    yte_hat = np.polyval(coef, xte)

    #compute metrics
    rmse_train = rmse(ytr, ytr_hat)
    rmse_test  = rmse(yte, yte_hat)
    mae_train = mae(ytr, ytr_hat)
    mae_test  = mae(yte, yte_hat)

    rows.append({
        "region_code": reg,
        "region_name": reg_name,
        "degree": DEGREE,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "mae_train": mae_train,
        "mae_test": mae_test})

    #accumulate for total
    Ytr_all.append(ytr);     Ytrhat_all.append(ytr_hat)
    Yte_all.append(yte);     Ytehat_all.append(yte_hat)

#total row (concatenate)
Ytr_all = np.concatenate(Ytr_all) if len(Ytr_all) else np.array([])
Ytrhat_all = np.concatenate(Ytrhat_all) if len(Ytrhat_all) else np.array([])
Yte_all = np.concatenate(Yte_all) if len(Yte_all) else np.array([])
Ytehat_all = np.concatenate(Ytehat_all) if len(Ytehat_all) else np.array([])

total_row = {
    "region_code": "TOTAL",
    "region_name": "Total",
    "degree": DEGREE,
    "n_train": int(len(Ytr_all)),
    "n_test": int(len(Yte_all)),
    "rmse_train": rmse(Ytr_all, Ytrhat_all),
    "rmse_test": rmse(Yte_all, Ytehat_all),
    "mae_train": mae(Ytr_all, Ytrhat_all),
    "mae_test": mae(Yte_all, Ytehat_all)}

summary = pd.DataFrame(rows + [total_row])

for i in ["rmse_train","rmse_test","mae_train","mae_test"]:
    summary[i] = pd.to_numeric(summary[i], errors="coerce").round(6)

save_path = os.path.abspath(OUT_SUMMARY_CSV)
summary.to_csv(save_path, index=False)