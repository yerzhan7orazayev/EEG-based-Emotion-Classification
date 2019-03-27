import os

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import layers
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

path = '/home/yerzhan/Desktop/Fall 2018/Project/'
os.chdir(path)

filename = 'data_python/spect_data_512.npz'
loaded = np.load(filename)

spect_data = loaded_1['data']
label = loaded['label']

spect_data = spect_data.reshape(19200, 17, 21, 1)

X_train_val, X_test, y_train_val, y_test = train_test_split(spect_data, label, test_size = 0.2, random_state = 42)

filepath = 'codes/model_CNN_512_v.hdf5'
callbacklist = [EarlyStopping(monitor='val_acc', mode = 'max', patience=10,),
                ModelCheckpoint(filepath=filepath, monitor='val_acc', save_best_only=True, mode = 'max')]

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(17, 21, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
history = model.fit(X_train_val, y_train_val, validation_split = 0.2, batch_size = 512, epochs = 200, verbose = 1)

# plot performance for each epoch
plt.rcParams.update({'font.size': 12})
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
   
plotTitle = 'Training and validation accuracy for Valence label (4s)'
plt.title(plotTitle)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy'], loc='best')

figureName = 'Plots/CNN_512_v.eps'
plt.savefig(figureName)

scores = model.evaluate(X_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))