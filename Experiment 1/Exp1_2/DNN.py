import pandas as pd
import numpy as np
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# 1. Load data
df=pd.read_parquet("Sparkov_data.parquet")
features = ['gender', 'city', 'state', 'lat', 'amt', 'category', 'transaction hour']
X = df[features].values
y = df['is_fraud'].values

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4. Define TPE search space
space = {
    'num_layers':    hp.quniform('num_layers', 2, 6, 1),
    'units':         hp.loguniform('units', np.log(32),  np.log(512)),
    'activation':    hp.choice('activation', ['relu', 'tanh', 'sigmoid', 'leaky_relu']),
    'dropout_rate':  hp.uniform('dropout_rate', 0.0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'batch_size':    hp.choice('batch_size', [16, 32, 64, 128]),
    'optimizer':     hp.choice('optimizer', ['sgd', 'adam', 'rmsprop']),
    'l2_reg':        hp.loguniform('l2_reg', np.log(1e-8), np.log(1e-3)),
}

# 5. Objective function (uses validation_split=0.2)
def objective(params):
    num_layers   = int(params['num_layers'])
    units        = int(params['units'])
    activation   = params['activation']
    dropout_rate = params['dropout_rate']
    lr           = params['learning_rate']
    l2_reg       = params['l2_reg']
    batch_size   = params['batch_size']
    opt_name     = params['optimizer']
    
    # Build model
    model = Sequential()
    # Input layer
    if activation == 'leaky_relu':
        model.add(Dense(units, kernel_regularizer=l2(l2_reg), input_shape=(X_train.shape[1],)))
        model.add(LeakyReLU())
    else:
        model.add(Dense(units, activation=activation,
                        kernel_regularizer=l2(l2_reg),
                        input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    # Hidden layers
    for _ in range(num_layers - 1):
        if activation == 'leaky_relu':
            model.add(Dense(units, kernel_regularizer=l2(l2_reg)))
            model.add(LeakyReLU())
        else:
            model.add(Dense(units, activation=activation,
                            kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimizer
    if opt_name == 'sgd':
        optimizer = SGD(learning_rate=lr)
    elif opt_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = RMSprop(learning_rate=lr)
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )
    # Use final validation accuracy as objective
    val_acc = history.history['val_accuracy'][-1]
    return {'loss': -val_acc, 'status': STATUS_OK}

# 6. Run hyperparameter optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

# 7. Rebuild model with best hyperparameters
# Map indices back to actual choices
activation_list = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
batch_size_list = [16, 32, 64, 128]
optimizer_list  = ['sgd', 'adam', 'rmsprop']

best_params = {
    'num_layers':   int(best['num_layers']),
    'units':        int(best['units']),
    'activation':   activation_list[best['activation']],
    'dropout_rate': best['dropout_rate'],
    'learning_rate':best['learning_rate'],
    'batch_size':   batch_size_list[best['batch_size']],
    'optimizer':    optimizer_list[best['optimizer']],
    'l2_reg':       best['l2_reg']
}

print("Best hyperparameters:", best_params)

# Build final model
model = Sequential()
act = best_params['activation']
# Input layer
if act == 'leaky_relu':
    model.add(Dense(best_params['units'], kernel_regularizer=l2(best_params['l2_reg']),
                    input_shape=(X_train.shape[1],)))
    model.add(LeakyReLU())
else:
    model.add(Dense(best_params['units'], activation=act,
                    kernel_regularizer=l2(best_params['l2_reg']),
                    input_shape=(X_train.shape[1],)))
model.add(Dropout(best_params['dropout_rate']))
# Hidden layers
for _ in range(best_params['num_layers'] - 1):
    if act == 'leaky_relu':
        model.add(Dense(best_params['units'], kernel_regularizer=l2(best_params['l2_reg'])))
        model.add(LeakyReLU())
    else:
        model.add(Dense(best_params['units'], activation=act,
                        kernel_regularizer=l2(best_params['l2_reg'])))
    model.add(Dropout(best_params['dropout_rate']))
# Output layer
model.add(Dense(1, activation='sigmoid'))

# Optimizer
opt = best_params['optimizer']
if opt == 'sgd':
    optimizer = SGD(learning_rate=best_params['learning_rate'])
elif opt == 'adam':
    optimizer = Adam(learning_rate=best_params['learning_rate'])
else:
    optimizer = RMSprop(learning_rate=best_params['learning_rate'])

model.compile(optimizer=optimizer, loss='binary_crossentropy')

# 8. Train on full training set (no validation split)
model.fit(X_train, y_train, epochs=10, batch_size=best_params['batch_size'], verbose=0)

# 9. Evaluate on the test set
y_pred_prob = model.predict(X_test, batch_size=best_params['batch_size']).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred)
auc   = roc_auc_score(y_test, y_pred_prob)
mcc   = matthews_corrcoef(y_test, y_pred)

print(f"Test Accuracy:  {acc:.4f}")
print(f"Precision:      {prec:.4f}")
print(f"Recall:         {rec:.4f}")
print(f"F1-score:       {f1:.4f}")
print(f"AUC:            {auc:.4f}")
print(f"MCC:            {mcc:.4f}")
