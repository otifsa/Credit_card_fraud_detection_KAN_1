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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# 1. Load data and select features
df=pd.read_parquet("Sparkov_data.parquet")
features = ['gender', 'city', 'state', 'lat', 'amt', 'category', 'transaction hour']
X = df[features].values
y = df['is_fraud'].values

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4. Reshape for GRU: each feature as a timestep
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test  = X_test.reshape(-1, X_test.shape[1], 1)

# 5. Define TPE search space for GRU
space = {
    'num_layers':    hp.quniform('num_layers', 1, 3, 1),
    'units':         hp.loguniform('units', np.log(32),  np.log(256)),
    'activation':    hp.choice('activation', ['relu', 'tanh', 'sigmoid']),
    'dropout_rate':  hp.uniform('dropout_rate', 0.0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
    'batch_size':    hp.choice('batch_size', [16, 32, 64]),
    'optimizer':     hp.choice('optimizer', ['sgd', 'adam', 'rmsprop']),
    'l2_reg':        hp.loguniform('l2_reg', np.log(1e-8), np.log(1e-3)),
}

# 6. Objective function
def objective(params):
    nl    = int(params['num_layers'])
    units = int(params['units'])
    act   = params['activation']
    dr    = params['dropout_rate']
    lr    = params['learning_rate']
    bs    = params['batch_size']
    optn  = params['optimizer']
    l2r   = params['l2_reg']
    
    model = Sequential()
    for i in range(nl):
        return_seq = (i < nl - 1)
        if i == 0:
            model.add(GRU(units,
                          activation=act,
                          return_sequences=return_seq,
                          kernel_regularizer=l2(l2r),
                          input_shape=(X_train.shape[1], 1)))
        else:
            model.add(GRU(units,
                          activation=act,
                          return_sequences=return_seq,
                          kernel_regularizer=l2(l2r)))
        model.add(Dropout(dr))
    
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = {
        'sgd':    SGD(learning_rate=lr),
        'adam':   Adam(learning_rate=lr),
        'rmsprop':RMSprop(learning_rate=lr)
    }[optn]
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=bs,
        validation_split=0.2,
        verbose=0
    )
    val_acc = history.history['val_accuracy'][-1]
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

# 8. Decode best hyperparameters
act_list   = ['relu', 'tanh', 'sigmoid']
batch_list = [16, 32, 64]
opt_list   = ['sgd', 'adam', 'rmsprop']

best_params = {
    'num_layers':   int(best['num_layers']),
    'units':        int(best['units']),
    'activation':   act_list[best['activation']],
    'dropout_rate': best['dropout_rate'],
    'learning_rate':best['learning_rate'],
    'batch_size':   batch_list[best['batch_size']],
    'optimizer':    opt_list[best['optimizer']],
    'l2_reg':       best['l2_reg']
}

print("Best hyperparameters:", best_params)

# 9. Rebuild final GRU model with best params
model = Sequential()
for i in range(best_params['num_layers']):
    return_seq = (i < best_params['num_layers'] - 1)
    if i == 0:
        model.add(GRU(best_params['units'],
                      activation=best_params['activation'],
                      return_sequences=return_seq,
                      kernel_regularizer=l2(best_params['l2_reg']),
                      input_shape=(X_train.shape[1], 1)))
    else:
        model.add(GRU(best_params['units'],
                      activation=best_params['activation'],
                      return_sequences=return_seq,
                      kernel_regularizer=l2(best_params['l2_reg'])))
    model.add(Dropout(best_params['dropout_rate']))

model.add(Dense(1, activation='sigmoid'))

optimizer = {
    'sgd':    SGD(learning_rate=best_params['learning_rate']),
    'adam':   Adam(learning_rate=best_params['learning_rate']),
    'rmsprop':RMSprop(learning_rate=best_params['learning_rate'])
}[best_params['optimizer']]

model.compile(optimizer=optimizer, loss='binary_crossentropy')

# 10. Train final model on full training data
model.fit(X_train, y_train,
          epochs=50,
          batch_size=best_params['batch_size'],
          verbose=0)

# 11. Evaluate on the test set
y_prob = model.predict(X_test, batch_size=best_params['batch_size']).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print(f"Test Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:      {precision_score(y_test, y_pred):.4f}")
print(f"Recall:         {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:       {f1_score(y_test, y_pred):.4f}")
print(f"AUC:            {roc_auc_score(y_test, y_prob):.4f}")
print(f"MCC:            {matthews_corrcoef(y_test, y_pred):.4f}")
