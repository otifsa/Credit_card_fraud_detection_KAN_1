
import pandas as pd
import numpy as np
import shap
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, BatchNormalization,
                                     LeakyReLU)
from tensorflow.keras.optimizers import Adam


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pykan import KAN


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)

# ========== DATA ==================================================
df=pd.read_parquet("Sparkov_data.parquet")
features = ['gender', 'city', 'state', 'lat', 'amt', 'category', 'transaction hour']
X = df[features].values
y = df['is_fraud'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.30, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# minority and majority masks
min_mask = y_train == 1
maj_mask = ~min_mask

X_min = X_train[min_mask]              # minority samples
n_min = X_min.shape[0]
n_maj = X_train[maj_mask].shape[0]

OVERSAMPLE_RATE = 0.30                 
N_SYNTH = int(OVERSAMPLE_RATE * n_maj) 

LATENT_DIM  = 32
GAN_LR      = 1e-3
GAN_STEPS   = 100                    

# ========== BUILD and TRAIN GAN =====================================
def build_generator():
    z = Input(shape=(LATENT_DIM,))
    x = Dense(128)(z); x = LeakyReLU(0.2)(x); x = BatchNormalization()(x)
    x = Dense(256)(x); x = LeakyReLU(0.2)(x); x = BatchNormalization()(x)
    x = Dense(512)(x); x = LeakyReLU(0.2)(x); x = BatchNormalization()(x)
    out = Dense(X_train.shape[1], activation="linear")(x)
    return Model(z, out, name="generator")

def build_discriminator():
    inp = Input(shape=(X_train.shape[1],))
    x = Dense(512)(inp); x = LeakyReLU(0.2)(x)
    x = Dense(256)(x); x = LeakyReLU(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out, name="discriminator")
    model.compile(optimizer=Adam(GAN_LR), loss="mse")
    return model

gen = build_generator()
disc = build_discriminator()

disc.trainable = False
z = Input(shape=(LATENT_DIM,))
valid = disc(gen(z))
gan = Model(z, valid, name="gan")
gan.compile(optimizer=Adam(GAN_LR), loss="mse")

batch_size = 64
ones  = np.ones((batch_size, 1), dtype=np.float32)
zeros = np.zeros_like(ones, dtype=np.float32)

for step in range(GAN_STEPS):
    # 1. Train discriminator
    idx     = np.random.randint(0, n_min, batch_size)
    real    = X_min[idx]
    noise   = np.random.normal(size=(batch_size, LATENT_DIM)).astype(np.float32)
    fake    = gen.predict(noise, verbose=0)
    disc.train_on_batch(real, ones)
    disc.train_on_batch(fake,  zeros)

    # Train generator
    noise = np.random.normal(size=(batch_size, LATENT_DIM)).astype(np.float32)
    gan.train_on_batch(noise, ones)

# ========== SYNTHETIC MINORITY CREATION ===========================
noise = np.random.normal(size=(N_SYNTH, LATENT_DIM)).astype(np.float32)
X_synth = gen.predict(noise, verbose=0)
y_synth = np.ones(N_SYNTH, dtype=np.int32)

# augmented training set
X_train_aug = np.vstack([X_train, X_synth])
y_train_aug = np.concatenate([y_train, y_synth])

# ========== KAN CLASSIFIER SETUP ==================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_kan(input_dim):
    return KAN(
        in_features=input_dim,
        num_layers=2,
        hidden_width=10,
        activation=F.silu,
        dropout=0.0,
        bspline_grid=20,
        bspline_order=3,
        grid_range=(-1, 1),
        shortcut=F.silu
    ).to(DEVICE)

kan = build_kan(X_train.shape[1])

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(kan.parameters(), lr=1e-3)
BATCH = 256
EPOCHS = 50

ds = TensorDataset(torch.from_numpy(X_train_aug).float(),
                   torch.from_numpy(y_train_aug).float().unsqueeze(1))
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

kan.train()
for _ in range(EPOCHS):
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = kan(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

# ========== EVALUATION METRICS ====================================
kan.eval()
with torch.no_grad():
    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
    logits   = kan(X_test_t)
    probs    = torch.sigmoid(logits).cpu().numpy().ravel()
    preds    = (probs >= 0.5).astype(int)

acc  = accuracy_score (y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec  = recall_score   (y_test, preds, zero_division=0)
f1   = f1_score       (y_test, preds, zero_division=0)
auc  = roc_auc_score  (y_test, probs)
mcc  = matthews_corrcoef(y_test, preds)

print(f"\nTest-set performance:")
print(f"  Accuracy : {acc :.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec :.4f}")
print(f"  F1-score : {f1 :.4f}")
print(f"  AUC      : {auc:.4f}")
print(f"  MCC      : {mcc :.4f}")

# ========== SHAP EXPLANATIONS =====================================
def kan_predict(data_np):
    data_t = torch.from_numpy(data_np.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(kan(data_t)).cpu().numpy()

shap.initjs()

background = shap.sample(X_train_aug, 100, random_state=0)
explainer  = shap.KernelExplainer(kan_predict, background)

sample_idx = np.random.choice(len(X_test), 300, replace=False)
shap_vals  = explainer.shap_values(X_test[sample_idx], nsamples=100)

mean_abs = np.abs(shap_vals).mean(axis=0)
mean_df  = pd.DataFrame({"feature": FEATURES, "mean_abs_SHAP": mean_abs})
mean_df  = mean_df.sort_values("mean_abs_SHAP", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(mean_df["feature"], mean_df["mean_abs_SHAP"])
plt.gca().invert_yaxis()
plt.title("Mean absolute SHAP value per feature")
plt.xlabel("|SHAP value|")
plt.tight_layout()
plt.show()
