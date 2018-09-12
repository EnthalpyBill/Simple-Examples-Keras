# 'Keras_classification' is used to classify Mnist data with neural network
# created by Bill

from tensorflow import keras as kr
import numpy as np

# download mnist data set
(x_train, y_train), (x_test, y_test) = kr.datasets.mnist.load_data()

# pre-process data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
y_train = kr.utils.to_categorical(y_train, num_classes = 10)
y_test = kr.utils.to_categorical(y_test, num_classes = 10)

# training parameters
times = 2 # training times
batch_size = 32
loss = 'categorical_crossentropy'
optimizer = 'adam'

# layer properties
hidden_layers = 1
units = 128
activation = 'relu'
activation_output = 'softmax'

# initialize model
model = kr.Sequential()

model.add(kr.layers.Dense(units = units, input_dim = 784)) # add 1st hidden layer (also input layer) 
model.add(kr.layers.Activation(activation)) # set activation function of 1st hidden layer

for i in range(1, hidden_layers):
	model.add(kr.layers.Dense(units = units)) # add the rest of hidden layers
	model.add(kr.layers.Activation(activation)) # set activation functions of the rest of hidden layers

model.add(kr.layers.Dense(units = 10)) # add output layer
model.add(kr.layers.Activation(activation_output)) # set activation function of output layer
          
model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy']) # set loss function and optimizer

# training
print('Begin Training')
model.fit(x_train, y_train, epochs = times, batch_size = batch_size, callbacks=[kr.callbacks.TensorBoard(log_dir='mytensorboard/3')])
print('End Training\n')

# testing
print('Begin Testing')
[cost, acc] = model.evaluate(x_test, y_test, batch_size = batch_size)
print('Loss:', cost, '\t', 'Accuracy:', acc)
print('End Testing')
