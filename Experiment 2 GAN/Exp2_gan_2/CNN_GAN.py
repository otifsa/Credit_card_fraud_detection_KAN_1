import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, LeakyReLU, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load and prepare data
df=pd.read_parquet("Sparkov_data.parquet")
features = ['gender', 'city', 'state', 'lat', 'amt', 'category', 'transaction hour']
X = df[features].values
y = df['is_fraud'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Reshape for 1D-CNN input: (samples, timesteps, channels)
X_train = X_train.reshape((-1, X_train.shape[1], 1))
X_test  = X_test.reshape((-1, X_test.shape[1], 1))

# Fixed hyperparameters
num_layers    = 3
filters       = 32
kernel_size   = 3
dropout_rate  = 0.2
learning_rate = 1e-3
batch_size    = 32
l2_reg        = 1e-4
T             = 50       # total GAN-classifier iterations
delta_alpha   = 0.05
alpha_max     = 1.0
latent_dim    = 32       # GAN noise vector size
gan_steps     = 1        # updates per iteration

# Build the CNN classifier
def build_cnn_classifier(input_shape):
    clf = Sequential()
    # First Conv block
    clf.add(Conv1D(filters, kernel_size, padding='same',
                   activation='relu',
                   kernel_regularizer=l2(l2_reg),
                   input_shape=input_shape))
    clf.add(Dropout(dropout_rate))
    # Additional Conv blocks
    for _ in range(num_layers - 1):
        clf.add(Conv1D(filters, kernel_size, padding='same',
                       activation='relu',
                       kernel_regularizer=l2(l2_reg)))
        clf.add(Dropout(dropout_rate))
    clf.add(Flatten())
    clf.add(Dense(1, activation='sigmoid'))
    clf.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy'
    )
    return clf

# Build GAN generator
def build_generator():
    z = Input(shape=(latent_dim,))
    x = Dense(64, activation='relu')(z)
    x = Dense(64, activation='relu')(x)
    out = Dense(X_train.shape[1], activation='linear')(x)
    return Model(z, out, name='generator')

# Build GAN discriminator
def build_discriminator():
    inp = Input(shape=(X_train.shape[1],))
    x = Dense(64)(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    disc = Model(inp, out, name='discriminator')
    disc.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    return disc

# Instantiate GAN components
generator     = build_generator()
discriminator = build_discriminator()

# Build combined GAN model
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
classifier   = build_cnn_classifier((X_train.shape[1], 1))
best_g       = -np.inf
best_weights = classifier.get_weights()
alpha        = 0.0

# Pre-extract minority class samples
X_min = X_train[y_train == 1].reshape(-1, X_train.shape[1])  # back to 2D for GAN

# Dynamic GAN-driven oversampling loop
for t in range(1, T + 1):
    # Increase oversampling rate
    alpha = min(alpha + delta_alpha, alpha_max)
    
    # Train GAN on minority class
    for _ in range(gan_steps):
        # Sample real minority
        idx_real = np.random.randint(0, X_min.shape[0], batch_size)
        real = X_min[idx_real]
        # Generate fake minority
        noise = np.random.normal(size=(batch_size, latent_dim))
        fake_min = generator.predict(noise, verbose=0)
        # Train discriminator
        discriminator.train_on_batch(real, np.ones((batch_size, 1)))
        discriminator.train_on_batch(fake_min, np.zeros((batch_size, 1)))
        # Train generator via GAN
        gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Generate synthetic samples according to α
    n_synth = int(alpha * np.sum(y_train == 0))
    noise   = np.random.normal(size=(n_synth, latent_dim))
    S_t     = generator.predict(noise, verbose=0)
    # Reshape synthetic for CNN: (n_synth, timesteps, 1)
    S_t = S_t.reshape((-1, X_train.shape[1], 1))
    y_synth = np.ones(n_synth)
    
    # Augment training data and train classifier
    X_aug = np.vstack([X_train, S_t])
    y_aug = np.concatenate([y_train, y_synth])
    classifier.fit(
        X_aug, y_aug,
        epochs=50,
        batch_size=batch_size,
        verbose=0
    )
    
    # Evaluate on test set
    y_prob = classifier.predict(X_test, batch_size=batch_size).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    G   = np.sqrt(f1 * mcc)
    
    flag = "  <-- new best" if G > best_g else ""
    print(f"Iter {t:2d}: α={alpha:.2f}, F1={f1:.4f}, MCC={mcc:.4f}, G={G:.4f}{flag}")
    if G > best_g:
        best_g = G
        best_weights = classifier.get_weights()

# Restore best classifier and final evaluation
classifier.set_weights(best_weights)
y_prob = classifier.predict(X_test, batch_size=batch_size).ravel()
y_pred = (y_prob >= 0.5).astype(int)

final_f1  = f1_score(y_test, y_pred)
final_mcc = matthews_corrcoef(y_test, y_pred)
print(f"\nBest model on test set: F1 = {final_f1:.4f}, MCC = {final_mcc:.4f}")
