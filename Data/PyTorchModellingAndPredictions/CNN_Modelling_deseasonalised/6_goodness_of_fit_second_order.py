import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

ROOT = "region_splits"
OUT_CSV_NAME = "test_temp_pred_actual.csv"
OUT_PNG_DIFF_NAME = "temp_vs_pred_minus_actual_second_order_fit.png"

regions = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

for reg in regions:
    reg_dir = os.path.join(ROOT, reg)
    csv_path = os.path.join(reg_dir, OUT_CSV_NAME)

    out_df = pd.read_csv(csv_path)

    # --- minimal fix: create pred_minus_actual if not present ---
    if "pred_minus_actual" not in out_df.columns:
        if {"predicted_excess", "actual_excess"}.issubset(out_df.columns):
            out_df["pred_minus_actual"] = (
                pd.to_numeric(out_df["predicted_excess"], errors="coerce")
                - pd.to_numeric(out_df["actual_excess"], errors="coerce")
            )
            out_df.to_csv(csv_path, index=False)  # persist for future runs
        else:
            raise KeyError(
                f"{csv_path} is missing 'pred_minus_actual' and also "
                f"does not contain both 'predicted_excess' and 'actual_excess'. "
                f"Available columns: {list(out_df.columns)}"
            )
    # ------------------------------------------------------------

    # assume columns already present and consistent
    xvals = pd.to_numeric(out_df["tempmax"], errors="coerce").to_numpy(float)
    yvals = pd.to_numeric(out_df["pred_minus_actual"], errors="coerce").to_numpy(float)

    #Plot predicted - actual + histogram
    fig, (ax_scatter, ax_bar) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0})
    ax_scatter.scatter(xvals, yvals, s=4, alpha=1.)
    ax_scatter.set_ylabel("Predicted - actual", labelpad=-1)
    ax_scatter.set_title(f"{reg} — Tempmax vs Predicted - Actual")

    #bootstrap 1000 second-order polynomial fits (y = a*x^2 + b*x + c)
    xgrid = np.linspace(xvals.min(), xvals.max(), 100)
    a_list = []
    b_list = []
    for i in range(1000):
        idx = np.random.randint(0, len(xvals), size=len(xvals))
        xb = xvals[idx]
        yb = yvals[idx]
        a, b, c = np.polyfit(xb, yb, 2)
        a_list.append(a)  # coefficient of x^2
        b_list.append(b)  # coefficient of x
        ygrid = a * xgrid**2 + b * xgrid + c
        ax_scatter.plot(xgrid, ygrid, linewidth=0.5, alpha=0.05, color="0.5")

    #best quadratic on all points, solid black
    a_best, b_best, c_best = np.polyfit(xvals, yvals, 2)
    ax_scatter.plot(xgrid, a_best * xgrid**2 + b_best * xgrid + c_best, linewidth=1.2, color="k")

    b_mean = float(np.mean(b_list))
    b_std = float(np.std(b_list, ddof=1))
    a_mean = float(np.mean(a_list))
    a_std = float(np.std(a_list, ddof=1))
    txt = f"a (x) = {b_mean:.4f} ± {b_std:.4f}\n" + f"b (x^2) = {a_mean:.4f} ± {a_std:.4f}"
    ax_scatter.text(0.98, 0.02, txt, transform=ax_scatter.transAxes, ha="right", va="bottom")

    counts, bins = np.histogram(xvals, bins=30)
    ax_bar.bar(bins[:-1], counts, width=np.diff(bins), align='edge')
    ax_bar.set_ylabel("Count")
    ax_bar.set_xlabel("Mean daily maximum temperature (°C)")

    fig.savefig(os.path.join(reg_dir, OUT_PNG_DIFF_NAME), dpi=350)
    plt.close(fig)
