import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load and prepare data
data = pd.read_csv('creditcard.csv')
features = [
    'V1', 'V4', 'V5', 'V8',
    'V10', 'V13', 'V14', 'V18',
    'V23', 'V26', 'Amount'
]
X = data[features].values
y = data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Fixed hyperparameters
num_layers    = 3
units         = 64
dropout_rate  = 0.2
learning_rate = 1e-3
batch_size    = 32
l2_reg        = 1e-4
T             = 50      
delta_alpha   = 0.05
alpha_max     = 1.0
latent_dim    = 32       
gan_steps     = 100     

# Build the classifier
def build_classifier(input_dim):
    clf = Sequential([
        Dense(units, activation='relu',
              kernel_regularizer=l2(l2_reg),
              input_shape=(input_dim,)),
        Dropout(dropout_rate),
        *[
            layer for _ in range(num_layers - 1)
            for layer in (
                Dense(units, activation='relu',
                      kernel_regularizer=l2(l2_reg)),
                Dropout(dropout_rate),
            )
        ],
        Dense(1, activation='sigmoid')
    ])
    clf.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy'
    )
    return clf

# Build GAN components based on your tables
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
    out = Dense(X_train.shape[1], activation='linear')(x)
    return Model(z, out, name='generator')

def build_discriminator():
    inp = Input(shape=(X_train.shape[1],))
    x = Dense(512)(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    disc = Model(inp, out, name='discriminator')
    disc.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'  # LSGAN loss
    )
    return disc

# Instantiate models
generator     = build_generator()
discriminator = build_discriminator()

# Build combined GAN
z = Input(shape=(latent_dim,))
fake = generator(z)
discriminator.trainable = False
validity = discriminator(fake)
gan = Model(z, validity, name='gan')
gan.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='mean_squared_error'
)

# Initialize classifier and tracking
classifier   = build_classifier(X_train.shape[1])
best_g       = -np.inf
best_weights = classifier.get_weights()
alpha        = 0.0

# Pre-extract minority samples
X_min = X_train[y_train == 1]

# Dynamic GAN-driven oversampling
for t in range(1, T + 1):
    alpha = min(alpha + delta_alpha, alpha_max)
    # Train the GAN
    for _ in range(gan_steps):
        idx_real = np.random.randint(0, X_min.shape[0], batch_size)
        real = X_min[idx_real]
        noise = np.random.normal(size=(batch_size, latent_dim))
        fake = generator.predict(noise, verbose=0)
        discriminator.train_on_batch(real, np.ones((batch_size, 1)))
        discriminator.train_on_batch(fake, np.zeros((batch_size, 1)))
        gan.train_on_batch(noise, np.ones((batch_size, 1)))
    # Generate synthetic samples
    n_synth = int(alpha * np.sum(y_train == 0))
    noise   = np.random.normal(size=(n_synth, latent_dim))
    S_t     = generator.predict(noise, verbose=0)
    y_synth = np.ones(n_synth)
    # Augment and train classifier
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
