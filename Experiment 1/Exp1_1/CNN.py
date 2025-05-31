import pandas as pd
import numpy as np
import tensorflow as tf
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
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
    Flatten,
    Dense,
    Dropout,
    Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# 1. Load and select features
data = pd.read_csv('creditcard.csv')
features = ['V1','V4','V5','V8','V10','V13','V14','V18','V23','V26','Amount']
X = data[features].values
y = data['Class'].values

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4. Reshape for Conv1D
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test  = X_test.reshape(-1, X_test.shape[1], 1)

# 5. Hyperopt search space
space = {
    'num_layers':    hp.quniform('num_layers', 2, 5, 1),
    'filters':       hp.loguniform('filters', np.log(16),  np.log(256)),
    'kernel_size':   hp.choice('kernel_size', [3, 5, 7]),
    'activation':    hp.choice('activation', ['relu', 'tanh', 'silu']),
    'dropout_rate':  hp.uniform('dropout_rate', 0.0, 0.5),
    'pooling_type':  hp.choice('pooling_type', ['max', 'avg']),
    'pool_size':     hp.choice('pool_size', [2, 3]),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'batch_size':    hp.choice('batch_size', [16, 32, 64, 128]),
    'optimizer':     hp.choice('optimizer', ['sgd', 'adam', 'rmsprop']),
    'l2_reg':        hp.loguniform('l2_reg', np.log(1e-8), np.log(1e-3)),
}

# 6. Objective function
def objective(params):
    nl = int(params['num_layers'])
    f  = int(params['filters'])
    ks = params['kernel_size']
    act = params['activation']
    dr = params['dropout_rate']
    pt = params['pooling_type']
    ps = params['pool_size']
    lr = params['learning_rate']
    l2r = params['l2_reg']
    bs = params['batch_size']
    opt_name = params['optimizer']

    model = Sequential()
    # first conv block
    if act == 'silu':
        model.add(Conv1D(f, ks, kernel_regularizer=l2(l2r),
                         input_shape=(X_train.shape[1],1), activation=None))
        model.add(Activation('swish'))
    else:
        model.add(Conv1D(f, ks, activation=act,
                         kernel_regularizer=l2(l2r),
                         input_shape=(X_train.shape[1],1)))
    # pooling + dropout
    if pt == 'max':
        model.add(MaxPooling1D(pool_size=ps))
    else:
        model.add(AveragePooling1D(pool_size=ps))
    model.add(Dropout(dr))

    # additional conv blocks
    for _ in range(nl-1):
        if act == 'silu':
            model.add(Conv1D(f, ks, kernel_regularizer=l2(l2r), activation=None))
            model.add(Activation('swish'))
        else:
            model.add(Conv1D(f, ks, activation=act, kernel_regularizer=l2(l2r)))
        if pt == 'max':
            model.add(MaxPooling1D(pool_size=ps))
        else:
            model.add(AveragePooling1D(pool_size=ps))
        model.add(Dropout(dr))

    # classifier head
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # optimizer
    optimizer = {
        'sgd': SGD(learning_rate=lr),
        'adam': Adam(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr)
    }[opt_name]

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

# 7. Run optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

# 8. Decode best parameters
kernel_list = [3, 5, 7]
act_list    = ['relu', 'tanh', 'silu']
pool_list   = ['max', 'avg']
batch_list  = [16, 32, 64, 128]
opt_list    = ['sgd', 'adam', 'rmsprop']

best_params = {
    'num_layers':   int(best['num_layers']),
    'filters':      int(best['filters']),
    'kernel_size':  kernel_list[best['kernel_size']],
    'activation':   act_list[best['activation']],
    'dropout_rate': best['dropout_rate'],
    'pooling_type': pool_list[best['pooling_type']],
    'pool_size':    best['pool_size'],
    'learning_rate':best['learning_rate'],
    'batch_size':   batch_list[best['batch_size']],
    'optimizer':    opt_list[best['optimizer']],
    'l2_reg':       best['l2_reg']
}

print("Best hyperparameters:", best_params)

# 9. Build final model
model = Sequential()
act = best_params['activation']
if act == 'silu':
    model.add(Conv1D(best_params['filters'], best_params['kernel_size'],
                     kernel_regularizer=l2(best_params['l2_reg']),
                     input_shape=(X_train.shape[1],1), activation=None))
    model.add(Activation('swish'))
else:
    model.add(Conv1D(best_params['filters'], best_params['kernel_size'],
                     activation=act,
                     kernel_regularizer=l2(best_params['l2_reg']),
                     input_shape=(X_train.shape[1],1)))
if best_params['pooling_type']=='max':
    model.add(MaxPooling1D(pool_size=best_params['pool_size']))
else:
    model.add(AveragePooling1D(pool_size=best_params['pool_size']))
model.add(Dropout(best_params['dropout_rate']))

for _ in range(best_params['num_layers']-1):
    if act == 'silu':
        model.add(Conv1D(best_params['filters'], best_params['kernel_size'],
                         kernel_regularizer=l2(best_params['l2_reg']), activation=None))
        model.add(Activation('swish'))
    else:
        model.add(Conv1D(best_params['filters'], best_params['kernel_size'],
                         activation=act, kernel_regularizer=l2(best_params['l2_reg'])))
    if best_params['pooling_type']=='max':
        model.add(MaxPooling1D(pool_size=best_params['pool_size']))
    else:
        model.add(AveragePooling1D(pool_size=best_params['pool_size']))
    model.add(Dropout(best_params['dropout_rate']))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
optimizer = {
    'sgd': SGD(learning_rate=best_params['learning_rate']),
    'adam': Adam(learning_rate=best_params['learning_rate']),
    'rmsprop': RMSprop(learning_rate=best_params['learning_rate'])
}[best_params['optimizer']]
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# 10. Train final model
model.fit(X_train, y_train,
          epochs=50,
          batch_size=best_params['batch_size'],
          verbose=0)

# 11. Evaluate
y_prob = model.predict(X_test, batch_size=best_params['batch_size']).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("Test Accuracy: ",  accuracy_score(y_test, y_pred))
print("Precision:     ",  precision_score(y_test, y_pred))
print("Recall:        ",  recall_score(y_test, y_pred))
print("F1-score:      ",  f1_score(y_test, y_pred))
print("AUC:           ",  roc_auc_score(y_test, y_prob))
print("MCC:           ",  matthews_corrcoef(y_test, y_pred))
