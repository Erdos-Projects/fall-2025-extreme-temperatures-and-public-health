import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#configure
INPUT_CSV = "region_weather_vs_excess_deaths.csv"
OUT_DIR = "region_splits"
TEST_FRAC = 0.2
SEED = 18

#weather feature columns to keep for training
FEAT_COLS = [
    "tempmax", "tempmin", "temp",
    "feelslikemax", "feelslikemin", "feelslike",
    "humidity", "precip", "precipprob", "precipcover",
    "snowdepth", "windspeed", "winddir",
    "sealevelpressure", "cloudcover", "visibility"]

df = pd.read_csv(INPUT_CSV)

#To plot at the end, we need tempmax and excess_deaths present
df_plot = df.dropna(subset=["tempmax", "excess_deaths"]).copy()

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

regions = sorted(df["region"].astype(str).unique().tolist())#get the regions

def save_scatter(df_xy: pd.DataFrame, out_png: str, title: str, bins: int = 30):
    #Top: scatter tempmax vs excess_deaths
    #Bottom: bar chart of counts by tempmax
    x = pd.to_numeric(df_xy["tempmax"], errors="coerce")
    y = pd.to_numeric(df_xy["excess_deaths"], errors="coerce")
    m = pd.notna(x) & pd.notna(y)
    x = x[m].to_numpy(float)
    y = y[m].to_numpy(float)

    fig, (ax_scatter, ax_bar) = plt.subplots(2, 1, figsize=(6, 5), sharex=True,gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0})

    #Top: scatter
    ax_scatter.scatter(x, y, s=8, alpha=1.)
    ax_scatter.set_ylabel("Excess deaths")
    ax_scatter.set_title(title)

    #Bottom: bar chart of counts by tempmax
    counts, edges = np.histogram(x, bins=bins)
    ax_bar.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
    ax_bar.set_ylabel("Count")
    ax_bar.set_xlabel("Mean daily maximum temperature (°C)")

    fig.savefig(out_png, dpi=350)
    plt.close(fig)

for reg in regions:
    #full feature set (per region) for saving to CSVs
    region_full = df[df["region"] == reg][FEAT_COLS + ["excess_deaths"]].reset_index(drop=True)

    #rows for plotting (require tempmax + target present)
    region_plot = df_plot[df_plot["region"] == reg][["tempmax", "excess_deaths"]].reset_index(drop=True)

    #row-level split indices based on g_full length
    idx = np.arange(len(region_full))
    rng.shuffle(idx)
    k = int(round(TEST_FRAC * len(region_full)))
    test_idx = idx[:k]
    train_idx = idx[k:]

    train_region_full = region_full.iloc[train_idx].reset_index(drop=True)
    test_region_full  = region_full.iloc[test_idx].reset_index(drop=True)

    #For plots, split g_plot separately for plotting consistency
    idx_p = np.arange(len(region_plot))
    rng.shuffle(idx_p)
    k_p = int(round(TEST_FRAC * len(region_plot)))
    test_idx_p = idx_p[:k_p]
    train_idx_p = idx_p[k_p:]
    train_region_plot = region_plot.iloc[train_idx_p].reset_index(drop=True)
    test_region_plot  = region_plot.iloc[test_idx_p].reset_index(drop=True)

    reg_dir = os.path.join(OUT_DIR, reg)
    os.makedirs(reg_dir, exist_ok=True)

    train_path = os.path.join(reg_dir, "train.csv")
    test_path  = os.path.join(reg_dir, "test.csv")
    train_region_full.to_csv(train_path, index=False)
    test_region_full.to_csv(test_path, index=False)

    #two-panel plots (tempmax vs excess_deaths + bottom bar counts)
    train_png = os.path.join(reg_dir, "train.png")
    test_png  = os.path.join(reg_dir, "test.png")
    save_scatter(train_region_plot, train_png, f"{reg} — Train")
    save_scatter(test_region_plot,  test_png,  f"{reg} — Test")