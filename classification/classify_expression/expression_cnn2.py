# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:09:54 2018

@author: zhylucky
"""
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#error in my computer
from PIL import *
import PIL.Image


# get data
with open("C:/Users/zhylucky/Desktop/python/classify_expression/fer2013/fer2013/less.csv") as f:
    content = f.readlines()
 
lines = np.array(content)
 
num_of_instances = lines.size - 1
print("number of instances: ",num_of_instances)

x_train, y_train, x_test, y_test = [], [], [], []
num_classes = 7
 
for i in range(1,num_of_instances):
 emotion, img, usage = lines[i].split(",")
 
 val = img.split(" ")
 pixels = np.array(val, 'float32')
 
 emotion = np_utils.to_categorical(emotion, num_classes)
 
 if 'Training' in usage:
     y_train.append(emotion)
     x_train.append(pixels)
 elif 'PublicTest' in usage:
     y_test.append(emotion)
     x_test.append(pixels)

# build a model
model = Sequential()
 
#1st convolution layer
model.add(Convolution2D(
        nb_filter = 64,
        nb_row = 5,
        nb_col = 5,
        border_mode = 'same',    # padding method
        input_shape = (1,        # channels
                       48,48),   # height & width
))  
model.add(Activation('relu'))
#1st pooling layer(MaximumPooling)
model.add(MaxPooling2D(
        pool_size = (5,5),
        strides = (2,2),
        border_mode = 'same',    # padding method    ### border_mode use?
))
 
#2nd convolution layer
model.add(Convolution2D(
        nb_filter = 64,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'same',    # padding method
))  
model.add(Activation('relu'))
 
#3rd convolution layer
model.add(Convolution2D(
        nb_filter = 64,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'same',    # padding method
))  
model.add(Activation('relu'))

#2nd pooling layer(AveragePooling)
model.add(AveragePooling2D(
        pool_size = (3,3),
        strides = (2,2),
        border_mode = 'same',    # padding method    ### border_mode use?
))
 
#4st convolution layer
model.add(Convolution2D(
        nb_filter = 128,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'same',    # padding method
))  
model.add(Activation('relu'))

#5st convolution layer
model.add(Convolution2D(
        nb_filter = 128,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'same',    # padding method
))  
model.add(Activation('relu'))

#3rd pooling layer(AveragePooling)
model.add(AveragePooling2D(
        pool_size = (3,3),
        strides = (2,2),
        border_mode = 'same',    # padding method    ### border_mode use?
))
 
model.add(Flatten())
 
#2 fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
 
model.add(Dense(num_classes, activation='softmax'))

#pre-process data
x_train = np.array(x_train,order='K')
x_train = x_train.reshape(x_train.shape[0],48,48)
x_train = x_train.reshape(-1, 1, 48, 48)/255
y_train = np.array(y_train,order='K')

x_test = np.array(x_test,order='K')
x_test = x_test.reshape(x_test.shape[0],48,48)
x_test = x_test.reshape(-1, 1, 48, 48)/255
y_test = np.array(y_test,order='K')

# train the model
gen = ImageDataGenerator(data_format = "channels_first" )
batch_size = 48
train_generator = gen.flow(x_train, y_train, batch_size = batch_size)
 
model.compile(loss='categorical_crossentropy'
    , optimizer = Adam(lr = 1e-4)
    , metrics=['accuracy'])

epochs = 1
model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1],'%')

# evaluate the model 
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1],'%')

# error
# test model using private picture
img = image.load_img("C:/Users/zhylucky/Desktop/python/classify_expression/pictures/happy.jpg", grayscale=True, target_size=(48, 48))
 
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
 
x /= 255
 
custom = model.predict(x)
emotion_analysis(custom[0])
 
x = np.array(x, 'float32')
x = x.reshape([48, 48]);
 
plt.gray()
plt.imshow(x)
plt.show()

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
 
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
 
    plt.show()