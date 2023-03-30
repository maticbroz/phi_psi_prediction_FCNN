import os
import pandas as pd
import tensorflow as tf
from keras import models, layers, optimizers
from keras import backend as K

# Constants
TRAIN_DATA_PATH = "../Inputs/train_data.csv"
VAL_DATA_PATH = "../Inputs/val_data.csv"
TEST_DATA_PATH = "../Inputs/test_data.csv"

# Import train and test data
train = pd.read_csv(TRAIN_DATA_PATH, delimiter=",")
val = pd.read_csv(VAL_DATA_PATH, delimiter=",")
test = pd.read_csv(TEST_DATA_PATH, delimiter=",")

# Inputs
X_train = train.iloc[:, 5:]
X_val = val.iloc[:, 5:]
X_test = test.iloc[:, 5:]

# Drop columns that contain "EEE". This way, the "EEE" is represented with a null vector.
X_train = X_train.drop(columns=X_train.filter(regex='EEE').columns)
X_val = X_val.drop(columns=X_val.filter(regex='EEE').columns)
X_test = X_test.drop(columns=X_test.filter(regex='EEE').columns)

# Features. "3" for phi, "4" for psi
y_train = train.iloc[:,3]
y_val = val.iloc[:,3]
y_test = test.iloc[:,3]

num_columns = X_train.shape[1]

def custom_loss(y_true, y_pred):
    D  = K.abs(y_pred - y_true)
    error = K.minimum(D, K.abs(360 - D))
    return error

# Model
model = models.Sequential()
model.add(layers.Dense(num_columns, input_dim=num_columns))
model.add(layers.Dense(num_columns, activation="relu"))
model.add(layers.Dense(num_columns // 2, activation="relu"))
model.add(layers.Dense(num_columns // 4, activation="relu"))
model.add(layers.Dense(num_columns // 8, activation="relu"))
model.add(layers.Dense(num_columns // 16, activation="relu"))
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer=optimizers.Adadelta(learning_rate=0.5),
                loss=custom_loss,
                )

callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2,factor=0.01, mode="auto", min_delta=0.1,cooldown=0,min_lr=10**-15)
history = model.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=50,batch_size=2**8, callbacks=[callback])

# Save the model
model.save("model.h5")

# Print the model summary
model.summary()
