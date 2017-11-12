'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 3

# the data, shuffled and split between train and test sets
# x_train = np.random.randn(50000, 784), np.random.randint(0, 10, 50000)

x_train = np.random.randn(50000, 784)
y_train = np.abs(np.around(np.mean(x_train, 1) * 100, decimals=0))
y_train = [y if 9 >= y >= 1 else 9 for y in y_train]
x_test = np.random.randn(10000, 784)
y_test = np.abs(np.around(np.mean(x_test, 1) * 100, decimals=0))
y_test = [y if 9 >= y >= 1 else 9 for y in y_test]
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='/Users/qiangwang/.keras/datasets/mnist.npz')
# x_train.shape (60000, 28, 28)
# x1 = x_train[0].reshape(1,784)
# x1[0][:100]
# y_train.shape (60000,)


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 去掉normalize以后效果很差
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
'''

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('/Users/qiangwang/.keras/datasets/mnist_model/all.h5')
with open('/Users/qiangwang/.keras/datasets/mnist_model/model.json', 'w') as out:
    out.write(model.to_json() + '\n')
model.save_weights('/Users/qiangwang/.keras/datasets/mnist_model/weight.h5')