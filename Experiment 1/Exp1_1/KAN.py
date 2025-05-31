import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# 1. Load & select features
data = pd.read_csv('creditcard.csv')
features = ['V1','V4','V5','V8','V10','V13','V14','V18','V23','V26','Amount']
X = data[features].values.astype(np.float32)
y = data['Class'].values.astype(np.float32)

# 2. Train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=0
)

# 3. Scale
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test       = scaler.transform(X_test)

# 4. Further split train → train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    stratify=y_train_full, test_size=0.2, random_state=0
)

# Convert to torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).unsqueeze(1).to(device)
X_val_t   = torch.from_numpy(X_val).to(device)
y_val_t   = torch.from_numpy(y_val).unsqueeze(1).to(device)
X_test_t  = torch.from_numpy(X_test).to(device)
y_test_t  = torch.from_numpy(y_test).unsqueeze(1).to(device)

# 5. Hyperopt search space for KAN
space = {
    'num_layers':    hp.quniform('num_layers', 1, 3, 1),
    'hidden_width':  hp.loguniform('hidden_width', np.log(32),  np.log(256)),
    'activation':    hp.choice('activation', ['relu','tanh','silu']),
    'dropout_rate':  hp.uniform('dropout_rate', 0.0, 0.3),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
    'batch_size':    hp.choice('batch_size', [32,64]),
    'optimizer':     hp.choice('optimizer', ['adam','lbfgs']),
    'l2_reg':        hp.loguniform('l2_reg', np.log(1e-8), np.log(1e-4)),
    'bspline_grid':  hp.quniform('bspline_grid', 10, 50, 1),
    'bspline_order': hp.choice('bspline_order', [2,3,4]),
    'shortcut':      hp.choice('shortcut', ['relu','tanh','silu']),
    # grid_range fixed to [-1,1]
}

# 6. Objective
def objective(params):
    # Cast hyperparameters
    nl        = int(params['num_layers'])
    width     = int(params['hidden_width'])
    act_str   = params['activation']
    dr        = params['dropout_rate']
    lr        = params['learning_rate']
    bs        = int(params['batch_size'])
    opt_str   = params['optimizer']
    l2r       = params['l2_reg']
    grid_pts  = int(params['bspline_grid'])
    grid_ord  = int(params['bspline_order'])
    sc_str    = params['shortcut']

    # 6a. Build KAN model
    from pykan import KAN
    activation = {'relu': F.relu, 'tanh': torch.tanh, 'silu': F.silu}[act_str]
    shortcut   = {'relu': F.relu, 'tanh': torch.tanh, 'silu': F.silu}[sc_str]

    model = KAN(
        in_features=X_train_t.size(1),
        num_layers=nl,
        hidden_width=width,
        activation=activation,
        dropout=dr,
        bspline_grid=grid_pts,
        bspline_order=grid_ord,
        grid_range=(-1.0,1.0),
        shortcut=shortcut
    ).to(device)

    # 6b. Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    if opt_str == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=l2r
        )
    else:  # LBFGS
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            weight_decay=l2r,
            max_iter=20
        )

    # 6c. DataLoader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

    # 6d. Training loop
    model.train()
    epochs = 50
    for epoch in range(epochs):
        for xb, yb in train_loader:
            if opt_str == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    # 6e. Validation accuracy
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_prob   = torch.sigmoid(val_logits).cpu().numpy()
        val_pred   = (val_prob >= 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_pred)

    return {'loss': -val_acc, 'status': STATUS_OK}

# 7. Run TPE optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

# 8. Decode best params
act_list    = ['relu','tanh','silu']
batch_list  = [32,64]
opt_list    = ['adam','lbfgs']
order_list  = [2,3,4]
shortcut_list = ['relu','tanh','silu']

best_params = {
    'num_layers':    int(best['num_layers']),
    'hidden_width':  int(best['hidden_width']),
    'activation':    act_list[best['activation']],
    'dropout_rate':  best['dropout_rate'],
    'learning_rate': best['learning_rate'],
    'batch_size':    batch_list[best['batch_size']],
    'optimizer':     opt_list[best['optimizer']],
    'l2_reg':        best['l2_reg'],
    'bspline_grid':  int(best['bspline_grid']),
    'bspline_order': order_list[best['bspline_order']],
    'shortcut':      shortcut_list[best['shortcut']],
}

print("Best hyperparameters:", best_params)

# 9. Rebuild and train final KAN on full training set
from pykan import KAN
activation = {'relu': F.relu, 'tanh': torch.tanh, 'silu': F.silu}[best_params['activation']]
shortcut   = {'relu': F.relu, 'tanh': torch.tanh, 'silu': F.silu}[best_params['shortcut']]

model = KAN(
    in_features=X_train_full.shape[1],
    num_layers=best_params['num_layers'],
    hidden_width=best_params['hidden_width'],
    activation=activation,
    dropout=best_params['dropout_rate'],
    bspline_grid=best_params['bspline_grid'],
    bspline_order=best_params['bspline_order'],
    grid_range=(-1.0,1.0),
    shortcut=shortcut
).to(device)

criterion = nn.BCEWithLogitsLoss()
if best_params['optimizer']=='adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['l2_reg']
    )
else:
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['l2_reg'],
        max_iter=20
    )

# Full train Loader
full_ds    = TensorDataset(torch.from_numpy(X_train_full).to(device),
                           torch.from_numpy(y_train_full).unsqueeze(1).to(device))
full_loader= DataLoader(full_ds, batch_size=best_params['batch_size'], shuffle=True)

model.train()
for epoch in range(50):
    for xb, yb in full_loader:
        if best_params['optimizer']=='lbfgs':
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

# 10. Evaluate on test set
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    prob   = torch.sigmoid(logits).cpu().numpy().ravel()
    pred   = (prob >= 0.5).astype(int)

print("Test Accuracy:",  accuracy_score(y_test, pred))
print("Precision:    ",  precision_score(y_test, pred))
print("Recall:       ",  recall_score(y_test, pred))
print("F1-score:     ",  f1_score(y_test, pred))
print("AUC:          ",  roc_auc_score(y_test, prob))
print("MCC:          ",  matthews_corrcoef(y_test, pred))
