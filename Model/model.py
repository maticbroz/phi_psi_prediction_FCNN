import time
import pandas as pd
from keras import models, layers, optimizers, backend as K
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
import optuna

# Load datasets
def load_data(path, skip_cols=5, drop_pattern='EEE', target_cols=(3,5)):
    data = pd.read_csv(path, delimiter=",")
    X = data.iloc[:, skip_cols:]
    X = X.drop(columns=X.filter(regex=drop_pattern).columns)
    y = data.iloc[:, target_cols[0]:target_cols[1]]
    return X, y

X_train, y_train = load_data("New_dataset/PISCES_train_OHE_2_5.csv")
X_val, y_val = load_data("New_dataset/PISCES_val_OHE_2_5.csv")
X_test, y_test = load_data("New_dataset/PISCES_test_OHE_2_5.csv")

def custom_loss(y_true, y_pred):
    D = K.abs(y_pred - y_true)
    return K.minimum(D, K.abs(360 - D))

def custom_loss_numpy(y_true, y_pred):
    D = np.abs(y_pred - y_true)
    return np.minimum(D, np.abs(360 - D))

def objective(trial):
    num_columns = X_train.shape[1]
    model = models.Sequential()
    model.add(layers.Dense(num_columns, input_dim=num_columns))

    num_layers = trial.suggest_int("num_layers", 1, 5)
    neurons_per_layer = trial.suggest_int("neurons_per_layer", 420 // 16, 420*2)

    for _ in range(num_layers):
        model.add(layers.Dense(neurons_per_layer, activation="relu"))

    model.add(layers.Dense(2, activation='linear'))
    model.compile(optimizer=optimizers.Adadelta(learning_rate=0.5), loss=custom_loss)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=2000, batch_size=2**8,
        callbacks=[EarlyStopping(monitor='val_loss', patience=4)]
    )

    y_pred = model.predict(X_test)
    test_loss_phi = round(np.mean(custom_loss_numpy(y_test.iloc[:, 0].to_numpy(), y_pred[:, 0])), 5)
    test_loss_psi = round(np.mean(custom_loss_numpy(y_test.iloc[:, 1].to_numpy(), y_pred[:, 1])), 5)

    return test_loss_phi + test_loss_psi

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best Hyperparameters: ", study.best_params)
print("Best Loss: ", study.best_value)
