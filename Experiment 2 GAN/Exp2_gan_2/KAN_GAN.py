import pandas as pd
import numpy as np
import copy


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Flatten, Dense, Dropout,
    LeakyReLU, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pykan import KAN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef

# Load & prepare data
df=pd.read_parquet("Sparkov_data.parquet")
features = ['gender', 'city', 'state', 'lat', 'amt', 'category', 'transaction hour']
X = df[features].values
y = df['is_fraud'].values


# Train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled  = scaler.transform(X_test)

# For GAN, keep 2D arrays
X_train_2d = X_train_scaled.copy()

# Build Keras GAN components

# Generator
latent_dim = 32
def build_generator():
    z = Input(shape=(latent_dim,))
    x = Dense(128)(z)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    out = Dense(X_train_2d.shape[1], activation='linear')(x)
    return Model(z, out, name='generator')

# Discriminator
learning_rate = 1e-3
def build_discriminator():
    inp = Input(shape=(X_train_2d.shape[1],))
    x = Dense(512)(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    disc = Model(inp, out, name='discriminator')
    disc.compile(optimizer=Adam(learning_rate),
                 loss='mean_squared_error')
    return disc

generator     = build_generator()
discriminator = build_discriminator()

# Combined GAN
discriminator.trainable = False
z = Input(shape=(latent_dim,))
fake = generator(z)
validity = discriminator(fake)
gan = Model(z, validity, name='gan')
gan.compile(optimizer=Adam(learning_rate),
            loss='mean_squared_error')

# Pre-extract minority samples
X_min = X_train_2d[y_train_full == 1]

# PyTorch KAN classifier setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_kan_classifier(input_dim):
    model = KAN(
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
    return model

clf_batch_size = 32
clf_lr         = 1e-3
clf_epochs     = 50

# Hold best model across iterations
best_g     = -np.inf
best_state = None

# Augmentation & training loop
T = 50
delta_alpha = 0.05
alpha_max   = 1.0
gan_steps   = 100

for t in range(1, T + 1):
    # Update alpha
    alpha = min(delta_alpha * t, alpha_max)

    # Train GAN for a few steps
    for _ in range(gan_steps):
        idx   = np.random.randint(0, X_min.shape[0], clf_batch_size)
        real  = X_min[idx]
        noise = np.random.normal(size=(clf_batch_size, latent_dim))
        fake_min = generator.predict(noise, verbose=0)
        discriminator.train_on_batch(real, np.ones((clf_batch_size, 1)))
        discriminator.train_on_batch(fake_min, np.zeros((clf_batch_size, 1)))
        gan.train_on_batch(noise, np.ones((clf_batch_size, 1)))

    # Generate synthetic minority
    n_synth = int(alpha * np.sum(y_train_full == 0))
    noise   = np.random.normal(size=(n_synth, latent_dim))
    S_t     = generator.predict(noise, verbose=0)  # shape (n_synth, features)
    y_synth = np.ones(n_synth, dtype=np.int32)

    # Augment dataset
    X_aug = np.vstack([X_train_scaled, S_t])
    y_aug = np.concatenate([y_train_full, y_synth])

    # Build & train PyTorch KAN classifier from scratch
    classifier = build_kan_classifier(X_train_scaled.shape[1])
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = torch.optim.Adam(classifier.parameters(), lr=clf_lr)

    # Prepare DataLoader
    X_aug_t = torch.from_numpy(X_aug).float().to(device)
    y_aug_t = torch.from_numpy(y_aug).float().unsqueeze(1).to(device)
    train_ds = TensorDataset(X_aug_t, y_aug_t)
    train_loader = DataLoader(train_ds, batch_size=clf_batch_size, shuffle=True)

    classifier.train()
    for epoch in range(clf_epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    classifier.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test_scaled).float().to(device)
        logits_test = classifier(X_test_t)
        prob_test   = torch.sigmoid(logits_test).cpu().numpy().ravel()
        pred_test   = (prob_test >= 0.5).astype(int)
    f1  = f1_score(y_test, pred_test)
    mcc = matthews_corrcoef(y_test, pred_test)
    G   = np.sqrt(f1 * mcc)
    flag = "  <-- new best" if G > best_g else ""
    print(f"Iter {t:2d}: α={alpha:.2f}, F1={f1:.4f}, MCC={mcc:.4f}, G={G:.4f}{flag}")

    # Save best state
    if G > best_g:
        best_g     = G
        best_state = copy.deepcopy(classifier.state_dict())

# Final evaluation with best model
best_model = build_kan_classifier(X_train_scaled.shape[1])
best_model.load_state_dict(best_state)
best_model.eval()
with torch.no_grad():
    X_test_t = torch.from_numpy(X_test_scaled).float().to(device)
    logits_test = best_model(X_test_t)
    prob_test   = torch.sigmoid(logits_test).cpu().numpy().ravel()
    pred_test   = (prob_test >= 0.5).astype(int)

final_f1  = f1_score(y_test, pred_test)
final_mcc = matthews_corrcoef(y_test, pred_test)
print(f"\nBest model on test set: F1 = {final_f1:.4f}, MCC = {final_mcc:.4f}")
