# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:19:09 2017

@author: B51427
"""
import csv
import cv2
import math
import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Dense, Conv2D, Flatten, MaxPooling2D, Activation, Cropping2D, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def generator(X_data, y_data, batch_size=128):
    while 1:
        for offset in range(0, len(X_data), batch_size):
            end=offset+batch_size
            X_batch=X_data[offset:end]
            y_batch=y_data[offset:end]
        yield shuffle(X_batch, y_batch)

lines=[]

with open('C:/CarND/behavioral_cloning_training_data/data/driving_log.csv', 'r') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
       
images=[]
angles=[]
delta=0.2

for line in lines:
    source_path = line[0]
    image_center=cv2.imread(source_path)
    
    source_path = line[1]
    image_left=cv2.imread(source_path)

    source_path = line[2]
    image_right=cv2.imread(source_path)
    
    image_center_flipped=np.fliplr(image_center)
    
    #add only center images to dataset for training
    images.append(image_center)
    images.append(image_center_flipped)
    #add all 3 camera images to dataset for training
    #images.extend(image_center, image_left, image_right)
    
    angle_center = float(line[3])
    angle_left = angle_center + delta
    angle_right = angle_center - delta
    
    #add only center image angle to dataset for training
    angles.append(angle_center)
    angles.append(angle_center*-1)
    #add all 3 angles to dataset for training
    #angles.extend(angle_center, angle_left, angle_right)

X_data=np.array(images)
y_data=np.array(angles)

X_data, y_data=shuffle(X_data, y_data)
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2)

batch_size=128

train_generator=generator(X_train, y_train, batch_size)
valid_generator=generator(X_valid, y_valid, batch_size)

#Here is the definition of neural network
model=Sequential()
model.add(Lambda(lambda x:x/128-1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(6,(5,5), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(12,(5,5), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(24,(5,5), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(240))
model.add(Activation('relu'))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

history_object = model.fit_generator(train_generator, steps_per_epoch =
    math.ceil(len(X_train)/batch_size), validation_data = 
    valid_generator,
    nb_val_samples = math.ceil(len(X_valid)/batch_size), 
    nb_epoch=10, verbose=1)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('graph.jpg')