import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

ROOT = "region_splits"
OUT_CSV_NAME = "test_temp_pred_actual.csv"
OUT_PNG_DIFF_NAME = "temp_vs_pred_minus_actual_straight_line_fit.png"

regions = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

for reg in regions:
    reg_dir = os.path.join(ROOT, reg)
    csv_path = os.path.join(reg_dir, OUT_CSV_NAME)

    out_df = pd.read_csv(csv_path)

    # --- minimal fix: create pred_minus_actual if not present ---
    if "pred_minus_actual" not in out_df.columns:
        if {"predicted_deaths", "actual_deaths"}.issubset(out_df.columns):
            out_df["pred_minus_actual"] = (
                pd.to_numeric(out_df["predicted_deaths"], errors="coerce")
                - pd.to_numeric(out_df["actual_deaths"], errors="coerce")
            )
            # persist so future runs don't error
            out_df.to_csv(csv_path, index=False)
        else:
            raise KeyError(
                f"{csv_path} is missing 'pred_minus_actual' and also "
                f"does not contain both 'predicted_deaths' and 'actual_deaths'. "
                f"Available columns: {list(out_df.columns)}"
            )
    # ------------------------------------------------------------

    xvals = pd.to_numeric(out_df["tempmax"], errors="coerce").to_numpy(float)
    yvals = pd.to_numeric(out_df["pred_minus_actual"], errors="coerce").to_numpy(float)

    # Plot predicted - actual + histogram
    fig, (ax_scatter, ax_bar) = plt.subplots(
        2, 1, figsize=(6, 5), sharex=True,
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0}
    )
    ax_scatter.scatter(xvals, yvals, s=4, alpha=1.)
    ax_scatter.set_ylabel("Predicted - actual", labelpad=-1)
    ax_scatter.set_title(f"{reg} — Tempmax vs Predicted - Actual")

    # bootstrap 1000 straight-line best fits (y = m*x + b) — grey lines, best in black
    xline = np.array([xvals.min(), xvals.max()])
    slopes = []
    for i in range(1000):
        idx = np.random.randint(0, len(xvals), size=len(xvals))
        xb = xvals[idx]
        yb = yvals[idx]
        m, b = np.polyfit(xb, yb, 1)
        slopes.append(m)
        yline = m * xline + b
        ax_scatter.plot(xline, yline, linewidth=0.5, alpha=0.05, color="0.5")

    # best fit on all points, solid black
    m_best, b_best = np.polyfit(xvals, yvals, 1)
    ax_scatter.plot(xline, m_best * xline + b_best, linewidth=1.2, color="k")

    m_mean = float(np.mean(slopes))
    m_std = float(np.std(slopes, ddof=1))
    txt = f"slope = {m_mean:.4f} ± {m_std:.4f}"
    ax_scatter.text(0.98, 0.02, txt, transform=ax_scatter.transAxes, ha="right", va="bottom")

    counts, bins = np.histogram(xvals, bins=30)
    ax_bar.bar(bins[:-1], counts, width=np.diff(bins), align='edge')
    ax_bar.set_ylabel("Count")
    ax_bar.set_xlabel("Mean daily maximum temperature (°C)")

    fig.savefig(os.path.join(reg_dir, OUT_PNG_DIFF_NAME), dpi=350)
    plt.close(fig)
