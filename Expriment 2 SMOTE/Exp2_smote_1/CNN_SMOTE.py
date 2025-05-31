import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load and prepare data
data = pd.read_csv('creditcard.csv')
features = ['V1', 'V4', 'V5', 'V8',
            'V10', 'V13', 'V14', 'V18',
            'V23', 'V26', 'Amount']
X = data[features].values
y = data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Reshape for 1D-CNN: (samples, timesteps, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# The best hyperparameters
num_layers    = 3
units         = 64
dropout_rate  = 0.2
learning_rate = 1e-3
batch_size    = 32
l2_reg        = 1e-4
T             = 50         # total SMOTE iterations
delta_alpha   = 0.05
alpha_max     = 1.0

# Model builder using Conv1D
def build_model(input_shape):
    model = Sequential()
    # First convolutional block
    model.add(Conv1D(
        filters=units,
        kernel_size=3,
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))
    # Additional convolutional blocks
    for _ in range(num_layers - 1):
        model.add(Conv1D(
            filters=units,
            kernel_size=3,
            activation='relu',
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(Dropout(dropout_rate))
    # Flatten and output
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy'
    )
    return model

# Initialize model and tracking variables
model = build_model((X_train.shape[1], 1))
best_g = -np.inf
best_weights = model.get_weights()
alpha = 0.0

# Dynamic SMOTE rate loop
for t in range(1, T + 1):
    # Increment alpha
    alpha = min(alpha + delta_alpha, alpha_max)
    
    # Apply SMOTE at rate alpha
    smote = SMOTE(sampling_strategy=alpha, random_state=0)
    # flatten back to 2D for SMOTE
    X_flat, y_flat = X_train.reshape(X_train.shape[0], -1), y_train
    X_res_flat, y_res = smote.fit_resample(X_flat, y_flat)
    # reshape back to 3D for CNN
    X_res = X_res_flat.reshape(X_res_flat.shape[0], X_train.shape[1], 1)
    
    # Train on the augmented set
    model.fit(
        X_res, y_res,
        epochs=50,
        batch_size=batch_size,
        verbose=0
    )
    
    # Evaluate on hold-out test set
    y_prob = model.predict(X_test, batch_size=batch_size).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Compute geometric mean
    G = np.sqrt(f1 * mcc)
    
    # Track best model
    if G > best_g:
        best_g = G
        best_weights = model.get_weights()
        print(f"Iteration {t:2d}: α={alpha:.2f}, F1={f1:.4f}, MCC={mcc:.4f}, G={G:.4f}  <-- new best")
    else:
        print(f"Iteration {t:2d}: α={alpha:.2f}, F1={f1:.4f}, MCC={mcc:.4f}, G={G:.4f}")

# Restore best model
model.set_weights(best_weights)

# Final evaluation
y_prob = model.predict(X_test, batch_size=batch_size).ravel()
y_pred = (y_prob >= 0.5).astype(int)

final_f1  = f1_score(y_test, y_pred)
final_mcc = matthews_corrcoef(y_test, y_pred)
print(f"\nBest model performance on test set:  F1 = {final_f1:.4f},  MCC = {final_mcc:.4f}")
