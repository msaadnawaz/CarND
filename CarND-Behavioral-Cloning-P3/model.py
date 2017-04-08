# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:19:09 2017

@author: B51427
"""
import csv
import cv2

import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Dense, Conv2D, Flatten, MaxPooling2D, Activation, Cropping2D, Dropout
from keras.models import Sequential

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

X_train=np.array(images)
y_train=np.array(angles)

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
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
#model.add(Activation('relu'))
#model.add(Dense())

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')