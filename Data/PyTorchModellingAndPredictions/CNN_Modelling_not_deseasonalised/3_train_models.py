import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#use best device for training
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch, "mps", None) and torch.mps.is_available():
    device = torch.device("mps")

###################################### SET-UP ############################################
ROOT = "region_splits"
EPOCHS = 10000 #cut off if training is complete earlier
LR = 2.5e-4
BATCH = 128
VAL_FRAC = 0.2
SEED = 18

NOISE_STD = 0.05 #add some noise to inputs before each training epoch
DROPOUT_P = 0.10 #add a lowish dropout

PATIENCE = 50 #early stopping fo training
MIN_DELTA = 1e-3

FEATS = ["tempmax","tempmin","snowdepth","humidity","windspeed","precip"]#,"windspeed","precip","snowdepth","humidity"] #features to train on
#the combination above provides the lowest MSE
##########################################################################################

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

def make_loaders(df: pd.DataFrame):
    # coerce + fill
    df = df.copy()
    for c in FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["snowdepth"] = df["snowdepth"].fillna(0.0)
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")

    df = df.dropna(subset=FEATS + ["deaths"]).reset_index(drop=True)

    #do training and validation split
    N = len(df)
    n_val = int(round(VAL_FRAC * N))
    idx = np.arange(N)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:] if n_val < N else idx

    tr = df.iloc[tr_idx].reset_index(drop=True)
    va = df.iloc[val_idx].reset_index(drop=True)

    #standardize X on TRAIN
    mu_x = tr[FEATS].mean()
    sd_x = tr[FEATS].std().replace(0, 1.0)

    Xtr = ((tr[FEATS] - mu_x) / sd_x).to_numpy(float)
    Xva = ((va[FEATS] - mu_x) / sd_x).to_numpy(float) if len(va) else np.empty((0, len(FEATS)))

    #standardize y on TRAIN
    mu_y = tr["deaths"].mean()
    sd_y = tr["deaths"].std()
    sd_y = 1.0 if (sd_y == 0 or np.isnan(sd_y)) else sd_y

    ytr = ((tr["deaths"] - mu_y) / sd_y).to_numpy(float).reshape(-1,1)
    yva = ((va["deaths"] - mu_y) / sd_y).to_numpy(float).reshape(-1,1) if len(va) else np.empty((0,1))

    tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                         torch.tensor(ytr, dtype=torch.float32)),
                           batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                                         torch.tensor(yva, dtype=torch.float32)),
                           batch_size=BATCH, shuffle=False)

    return mu_x, sd_x, mu_y, sd_y, tr_loader, va_loader

#fairly simple NN (may even be a bit too complex for the data)
class MLP(nn.Module):
    def __init__(self, in_feats:int, p=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

def train_one_region(region_dir: str):
    #Use TRAIN split to fit the model and compute normalization stats
    train_path = os.path.join(region_dir, "train.csv")

    df = pd.read_csv(train_path)

    mu_x, sd_x, mu_y, sd_y, tr_loader, va_loader = make_loaders(df)

    model = MLP(in_feats=len(FEATS), p=DROPOUT_P).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    bad_epochs = 0

    for ep in range(1, EPOCHS + 1):
        #train
        model.train()
        total, n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)

            #add fresh Gaussian noise each epoch to improve training
            if NOISE_STD > 0:
                xb = xb + NOISE_STD * torch.randn_like(xb)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * yb.size(0)
            n += yb.size(0)
        train_mse = total / max(1, n)

        #validate
        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vtotal += loss_fn(pred, yb).item() * yb.size(0)
                vn += yb.size(0)
        val_mse = (vtotal / max(1, vn)) if vn > 0 else float("nan")

        improved = (val_mse < best_val - MIN_DELTA) if np.isfinite(val_mse) else False
        if improved: #save best
            best_val = val_mse
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(region_dir, "model.pt"))
        else:
            bad_epochs += 1

        if ep % 25 == 0 or ep == 1:
            print(f"epoch {ep:4d}, train MSE is {train_mse:.4f}, val MSE is {val_mse:.4f}")

        if bad_epochs >= PATIENCE:
            print(f"Stopping at epoch {ep}")
            break

    if not os.path.isfile(os.path.join(region_dir, "model.pt")):
        torch.save(model.state_dict(), os.path.join(region_dir, "model.pt"))

    #Save training feature order + stats for test-time
    np.savez(os.path.join(region_dir, "preprocess.npz"),
             mu=mu_x.to_numpy(), sd=sd_x.to_numpy(),
             feat_cols=np.array(FEATS, dtype=object),
             y_mu=np.array([mu_y]), y_sd=np.array([sd_y]))

    print(f"Saved model")

if not os.path.isdir(ROOT):
	print("Run scripts 1 and 2 first")

regions = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

# run training functions for each region
for reg in regions:
    print(f"Region: {reg}")
    train_one_region(os.path.join(ROOT, reg))
