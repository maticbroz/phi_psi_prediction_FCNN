import pandas as pd
from tensorflow import keras
from keras import layers, optimizers, callbacks
from keras import backend as K
import keras

# Import train and test data
train = pd.read_csv("../Inputs/train_data.csv", delimiter=",")
test = pd.read_csv("../Inputs/test_data.csv", delimiter=",")

X_train = train.iloc[:, 4:]
X_test = test.iloc[:, 4:]

y_train = train.iloc[:,2]
y_test = test.iloc[:,2]

def my_loss_fn(y_true, y_pred):
    D  = K.abs(y_true - y_pred)
    AE = K.minimum(D, K.abs(360 - D))
    return AE

network = keras.Sequential()
network.add(layers.Dense(X_train.shape[1], activation='relu'))
network.add(layers.Dense(X_train.shape[1] * 20, activation='relu'))
network.add(layers.Dense(X_train.shape[1] * 20, activation='relu'))
network.add(layers.Dense(X_train.shape[1], activation='relu'))
network.add(layers.Dense(1))

network.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=my_loss_fn)

callback = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='auto')

network.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop, callback])
