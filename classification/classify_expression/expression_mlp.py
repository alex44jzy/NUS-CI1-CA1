import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
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
x_train = x_train.reshape(x_train.shape[0], -1) / 255.   # normalize
y_train = np.array(y_train,order='K')

x_test = np.array(x_test,order='K')
x_test = x_test.reshape(x_test.shape[0], -1) / 255.   # normalize
y_test = np.array(y_test,order='K')


# build your neural net
model = Sequential([
    Dense(32, input_dim=2304),
    Activation('relu'),
    Dense(7),
    Activation('softmax'),
])

# define optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# train the model
model.fit(x_train, y_train, epochs=2, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)      # test accuracy:  0.9509