# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:19:09 2017

@author: B51427
"""
import csv
import cv2




import pickle
import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Dense, Convolution2D, Flatten, Maxpooling2D, Activation
from keras.models import Sequential

X_train=np.array(images)
y_train=np.array(angles)

#Here is the definition of neural network
model=Sequential()
model.add(Convolution2D())
model.add(Maxpooling2D())
model.add(Convolution2D())
model.add(Flatten())
model.add(Dense())
model.add(Activation('relu'))
model.add(Dense())
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save(model.h5)