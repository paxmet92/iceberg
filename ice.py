from __future__ import print_function
import numpy as np
import pandas as pd

import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection

import keras
print(keras.__version__)

import ijson

train = pd.read_json("train.json")

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_full = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_full = train['is_iceberg'].values
X, Xt, y, yt = sklearn.model_selection.train_test_split(X_full, y_full, test_size=0.2, random_state=0)

def getModel():
    #Building the model
    gmodel=keras.models.Sequential()
    #Conv Layer 1
    gmodel.add(keras.layers.Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 2)))
    gmodel.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(keras.layers.Dropout(0.2))

    #Conv Layer 2
    gmodel.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(keras.layers.Dropout(0.2))

    #Conv Layer 3
    gmodel.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(keras.layers.Dropout(0.2))

    #Conv Layer 4
    gmodel.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(keras.layers.Dropout(0.2))

    #Flatten the data for upcoming dense layers
    gmodel.add(keras.layers.Flatten())

    #Dense Layers
    gmodel.add(keras.layers.Dense(512))
    gmodel.add(keras.layers.Activation('relu'))
    gmodel.add(keras.layers.Dropout(0.2))

    #Dense Layer 2
    gmodel.add(keras.layers.Dense(256))
    gmodel.add(keras.layers.Activation('relu'))
    gmodel.add(keras.layers.Dropout(0.2))

    #Sigmoid Layer
    gmodel.add(keras.layers.Dense(1))
    gmodel.add(keras.layers.Activation('sigmoid'))

    mypotim=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel

def get_callbacks(filepath, patience=2):
    es = keras.callbacks.EarlyStopping('val_loss', patience=patience, mode="min")
    msave = keras.callbacks.ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]
file_path = ".ghmodel_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

gmodel=getModel()
gmodel.fit(X, y,
          batch_size=24,
          epochs=50,
          validation_data=(Xt,yt),
          verbose=1,
          callbacks=callbacks)