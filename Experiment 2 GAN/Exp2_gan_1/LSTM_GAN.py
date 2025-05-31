import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout,
    LeakyReLU, BatchNormalization, Reshape
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# ————————————————
# Load and prepare data
# ————————————————
data = pd.read_csv('creditcard.csv')
features = [
    'V1', 'V4', 'V5', 'V8',
    'V10', 'V13', 'V14', 'V18',
    'V23', 'V26', 'Amount'
]
X = data[features].values
y = data['Class'].values

X_train_2d, X_test_2d, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train_2d = scaler.fit_transform(X_train_2d)
X_test_2d  = scaler.transform(X_test_2d)

# reshape for LSTM
timesteps = X_train_2d.shape[1]  # 11
X_train = X_train_2d.reshape(-1, timesteps, 1)
X_test  = X_test_2d.reshape(-1, timesteps, 1)

# ————————————————
# Hyperparameters
# ————————————————
num_layers    = 2
units         = 128
dropout_rate  = 0.4
learning_rate = 1e-3
batch_size    = 128
l2_reg        = 1e-4

T             = 50
delta_alpha   = 0.05
alpha_max     = 1.0

latent_dim    = 32
gan_steps     = 100

# ————————————————
# Build the LSTM classifier
# ————————————————
def build_classifier(input_shape):
    clf = Sequential()
    clf.add(LSTM(
        units, activation='tanh',
        kernel_regularizer=l2(l2_reg),
        return_sequences=(num_layers > 1),
        input_shape=input_shape
    ))
    clf.add(Dropout(dropout_rate))
    for i in range(1, num_layers):
        is_last = (i == num_layers - 1)
        clf.add(LSTM(
            units, activation='tanh',
            kernel_regularizer=l2(l2_reg),
            return_sequences=not is_last
        ))
        clf.add(Dropout(dropout_rate))
    clf.add(Dense(1, activation='sigmoid'))
    clf.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy'
    )
    return clf

# ————————————————
# Build GAN generator (Table 13)
# ————————————————
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

    out = Dense(timesteps, activation='linear')(x)
    return Model(z, out, name='generator')

# ————————————————
# Build GAN discriminator (Table 14)
# ————————————————
def build_discriminator():
    inp = Input(shape=(timesteps,))
    x = Dense(512)(inp)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)

    out = Dense(1, activation='sigmoid')(x)
    disc = Model(inp, out, name='discriminator')
    disc.compile(
        optimizer=Adam(learning_rate),
        loss='mean_squared_error'  # LSGAN
    )
    return disc

# ————————————————
# Instantiate GAN & classifier
# ————————————————
generator     = build_generator()
discriminator = build_discriminator()

# Combined GAN
z = Input(shape=(latent_dim,))
fake = generator(z)
discriminator.trainable = False
validity = discriminator(fake)
gan = Model(z, validity, name='gan')
gan.compile(
    optimizer=Adam(learning_rate),
    loss='mean_squared_error'
)

classifier   = build_classifier((timesteps, 1))
best_g       = -np.inf
best_weights = classifier.get_weights()
alpha        = 0.0

# Pre-extract minority samples (2D)
X_min = X_train_2d[y_train == 1]

# ————————————————
# Dynamic GAN-driven oversampling
# ————————————————
for t in range(1, T + 1):
    alpha = min(alpha + delta_alpha, alpha_max)

    # Train GAN
    for _ in range(gan_steps):
        idx_real = np.random.randint(0, X_min.shape[0], batch_size)
        real = X_min[idx_real]
        noise = np.random.normal(size=(batch_size, latent_dim))
        fake_min = generator.predict(noise, verbose=0)

        discriminator.train_on_batch(real, np.ones((batch_size, 1)))
        discriminator.train_on_batch(fake_min, np.zeros((batch_size, 1)))
        gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Generate synthetic minority
    n_synth = int(alpha * np.sum(y_train == 0))
    noise   = np.random.normal(size=(n_synth, latent_dim))
    S_t_2d  = generator.predict(noise, verbose=0)
    S_t     = S_t_2d.reshape(-1, timesteps, 1)
    y_synth = np.ones(n_synth)

    # Augment & train classifier
    X_aug = np.vstack([X_train, S_t])
    y_aug = np.concatenate([y_train, y_synth])
    classifier.fit(
        X_aug, y_aug,
        epochs=50,
        batch_size=batch_size,
        verbose=0
    )

    # Evaluate
    y_prob = classifier.predict(X_test, batch_size=batch_size).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    G   = np.sqrt(f1 * mcc)

    flag = "  <-- new best" if G > best_g else ""
    print(f"Iter {t:2d}: α={alpha:.2f}, F1={f1:.4f}, MCC={mcc:.4f}, G={G:.4f}{flag}")
    if G > best_g:
        best_g       = G
        best_weights = classifier.get_weights()

# Final evaluation
classifier.set_weights(best_weights)
y_prob = classifier.predict(X_test, batch_size=batch_size).ravel()
y_pred = (y_prob >= 0.5).astype(int)
final_f1  = f1_score(y_test, y_pred)
final_mcc = matthews_corrcoef(y_test, y_pred)
print(f"\nBest model on test set: F1 = {final_f1:.4f}, MCC = {final_mcc:.4f}")
