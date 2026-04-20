"""
LoL Win Probability Model — Trainer
Trainiert ein Neural Network auf lol_dataset.csv und speichert lol_model.pt.

Requirements:
    pip install torch pandas numpy scikit-learn
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "lol_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "lol_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "lol_scaler.json")
FEAT_PATH   = os.path.join(BASE_DIR, "lol_features.json")

BATCH_SIZE = 256
EPOCHS     = 100
LR         = 1e-3
HIDDEN     = [256, 128, 64]
DROPOUT    = 0.3
VAL_SPLIT  = 0.15
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Feature Prep ───────────────────────────────────────────────────────────────

DROP_COLS = ["match_id", "own_team", "own_champ", "result"]

def prepare(df):
    df = df.copy()

    # pX_team droppen — immer gleich (p1-5 = blue, p6-10 = red), kein Signal
    team_cols  = [f"p{i}_team"  for i in range(1, 11)]
    champ_cols = [f"p{i}_champ" for i in range(1, 11)]
    drop = [c for c in DROP_COLS + team_cols + champ_cols if c in df.columns]

    y    = df["result"].values.astype(np.float32)
    X_df = df.drop(columns=drop)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X_df.values.astype(np.float32), y, list(X_df.columns)

# ── Scaler ─────────────────────────────────────────────────────────────────────

class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"mean": self.mean_.tolist(), "std": self.std_.tolist()}, f)

# ── Model ──────────────────────────────────────────────────────────────────────

class WinProbModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# ── Dataset ────────────────────────────────────────────────────────────────────

class SnapshotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Training ───────────────────────────────────────────────────────────────────

def train():
    print("═" * 58)
    print("  LoL Win Probability — Trainer")
    print("═" * 58)

    print(f"\n  Lade {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  ✓ {len(df)} Snapshots aus {df['match_id'].nunique()} Games")

    X, y, feature_names = prepare(df)
    print(f"  ✓ {X.shape[1]} Features | {int(y.sum())} Wins / {int((1-y).sum())} Losses")

    # Scaler fitten und speichern
    scaler = SimpleScaler()
    X = scaler.fit_transform(X)
    scaler.save(SCALER_PATH)
    print(f"  ✓ Scaler gespeichert: {SCALER_PATH}")

    with open(FEAT_PATH, "w") as f:
        json.dump(feature_names, f)
    print(f"  ✓ Features gespeichert: {FEAT_PATH}")

    # Train/Val Split nach match_id — kein Game in beiden Sets
    match_ids = df["match_id"].unique()
    n_val     = max(1, int(len(match_ids) * VAL_SPLIT))
    rng       = np.random.default_rng(SEED)
    val_ids   = set(rng.choice(match_ids, n_val, replace=False))

    train_mask = ~df["match_id"].isin(val_ids).values
    val_mask   =  df["match_id"].isin(val_ids).values

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    print(f"\n  Train: {len(X_train)} Snapshots | Val: {len(X_val)} Snapshots ({n_val} Games)")

    train_dl = DataLoader(SnapshotDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl   = DataLoader(SnapshotDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    model   = WinProbModel(X.shape[1], HIDDEN, DROPOUT).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.BCELoss()

    best_val_auc = 0
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val Acc':<10} {'Val AUC'}")
    print(f"  {'─'*8} {'─'*14} {'─'*14} {'─'*10} {'─'*8}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                probs = model(xb)
                val_loss += loss_fn(probs, yb).item() * len(yb)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        val_loss /= len(X_val)

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        val_acc    = accuracy_score(all_labels, (all_probs >= 0.5).astype(int)) * 100
        val_auc    = roc_auc_score(all_labels, all_probs)
        sched.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state": model.state_dict(),
                "input_dim":   X.shape[1],
                "hidden":      HIDDEN,
                "dropout":     DROPOUT,
            }, MODEL_PATH)
            saved = " ← saved"
        else:
            saved = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:<8} {train_loss:<14.4f} {val_loss:<14.4f} {val_acc:<9.1f}%  {val_auc:.4f}{saved}")

    print(f"\n{'═'*58}")
    print(f"  ✓ Bestes Modell: AUC {best_val_auc:.4f}")
    print(f"  ✓ Gespeichert:   {MODEL_PATH}")
    print(f"{'═'*58}\n")


# ── Inferenz (für LiveWinChance.py) ───────────────────────────────────────────

def predict(snapshot_dict: dict) -> float:
    with open(FEAT_PATH) as f:
        feature_names = json.load(f)

    with open(SCALER_PATH) as f:
        scaler_data = json.load(f)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std  = np.array(scaler_data["std"],  dtype=np.float32)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = WinProbModel(checkpoint["input_dim"], checkpoint["hidden"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    row = []
    for feat in feature_names:
        val = snapshot_dict.get(feat, 0)
        if isinstance(val, str):
            val = 1.0 if val == "blue" else 0.0
        row.append(float(val))

    X = np.array([row], dtype=np.float32)
    X = (X - mean) / std

    with torch.no_grad():
        return model(torch.tensor(X)).item()


if __name__ == "__main__":
    train()
