from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

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

plt.imshow(x_train[2].reshape((48,48)))
plt.show() 
    
#pre-process data
x_train = np.array(x_train,order='K')
x_train = x_train.reshape(x_train.shape[0],48,48)
x_train = x_train.reshape(-1, 1, 48, 48)/255
y_train = np.array(y_train,order='K')

x_test = np.array(x_test,order='K')
x_test = x_test.reshape(x_test.shape[0],48,48)
x_test = x_test.reshape(-1, 1, 48, 48)/255
y_test = np.array(y_test,order='K')
  
# draw the ordinal figure
plt.imshow(x_train[2].reshape((48,48)),cmap = 'Greys')
plt.show() 
#y_train = np_utils.to_categorical(y_train)    

# build CNN1
model = Sequential()

# Conv layer 1 output (32, 28, 28)
model.add(Convolution2D(
        nb_filter = 62,
        nb_row = 5,
        nb_col = 5,
        border_mode = 'same',    # padding method
        input_shape = (1,        # channels
                       48,48),   # height & width
))  
model.add(Activation('relu')) 

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'same',    # padding method 
))  

#Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, 5, border_mode = 'same'))
model.add(Activation('relu'))

#Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(pool_size = (2,2), 
                       border_mode = 'same'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136) , output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(7))
model.add(Activation('softmax'))

# define optimizer
adam = Adam(lr = 1e-4)

# We add matrics to get more results
model.compile(optimizer = adam,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

print ('Training ---------------')
# train the model
model.fit(x_train, y_train, nb_epoch = 2, batch_size =32)

print ('\nTesting --------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss:', loss)
print('\ntest accuracy:', accuracy)  # test accuracy: 

