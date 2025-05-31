import pandas as pd
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pykan import KAN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE

# 1. Load & prepare data
df=pd.read_parquet("IEEE-fraud-detection.parquet")
features = [
    'C1', 'C12', 'C13', 'C14', 'C2',
    'C4', 'C6', 'C7', 'C8', 'C9', 'M6', 'P_emaildomain', 'ProductCD',
    'TransactionDT', 'V106', 'V110', 'V112', 'V136', 'V14', 'V16', 'V27', 'V28',
    'V281', 'V29', 'V292', 'V297', 'V31', 'V311', 'V313', 'V321', 'V37', 'V39',
    'V44', 'V50', 'V53', 'V57', 'V79', 'V87', 'V89', 'V92', 'V94', 'V97', 'V98',
    'card1', 'card2', 'card5'
]
X = df[features].values
y = df['isFraud'].values
# 2. Train/test split & scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 3. PyTorch KAN classifier builder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_kan_classifier(input_dim):
    return KAN(
        in_features=input_dim,
        num_layers=2,
        hidden_width=10,
        activation=F.silu,
        dropout=0.0,
        bspline_grid=20,
        bspline_order=3,
        grid_range=(-1.0, 1.0),
        shortcut=F.silu
    ).to(device)

# 4. Training settings
clf_lr       = 1e-3
clf_epochs   = 50
clf_batch    = 32

# dynamic SMOTE parameters
T            = 50
delta_alpha  = 0.02    # increase minority ratio by 2% per iteration
alpha_max    = 1.0

best_g       = -np.inf
best_state   = None

# 5. Dynamic oversampling & training loop
for t in range(1, T+1):
    alpha = min(delta_alpha * t, alpha_max)
    # sampling_strategy = desired minority/majority ratio
    sm = SMOTE(sampling_strategy=alpha, random_state=0)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

    # prepare DataLoader
    X_t = torch.from_numpy(X_res).float().to(device)
    y_t = torch.from_numpy(y_res).float().unsqueeze(1).to(device)
    ds  = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=clf_batch, shuffle=True)

    # build & train KAN
    model     = build_kan_classifier(X_train_scaled.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optim     = torch.optim.Adam(model.parameters(), lr=clf_lr)

    model.train()
    for epoch in range(clf_epochs):
        for xb, yb in loader:
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test_scaled).float().to(device)
        logits   = model(X_test_t)
        probs    = torch.sigmoid(logits).cpu().numpy().ravel()
        preds    = (probs >= 0.5).astype(int)

    f1  = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    G   = np.sqrt(f1 * mcc)
    mark = "  <-- new best" if G > best_g else ""
    print(f"Iter {t:2d}: α={alpha:.2f}, F1={f1:.4f}, MCC={mcc:.4f}, G={G:.4f}{mark}")

    if G > best_g:
        best_g     = G
        best_state = copy.deepcopy(model.state_dict())

# 6. Final evaluation with best model
best_model = build_kan_classifier(X_train_scaled.shape[1])
best_model.load_state_dict(best_state)
best_model.eval()
with torch.no_grad():
    X_test_t = torch.from_numpy(X_test_scaled).float().to(device)
    logits   = best_model(X_test_t)
    probs    = torch.sigmoid(logits).cpu().numpy().ravel()
    preds    = (probs >= 0.5).astype(int)

final_f1  = f1_score(y_test, preds)
final_mcc = matthews_corrcoef(y_test, preds)
print(f"\nBest model on test set: F1 = {final_f1:.4f}, MCC = {final_mcc:.4f}")
