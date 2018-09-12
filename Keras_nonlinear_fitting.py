# 'Keras_nonlinear_fitting' is used to fit nonlinear data with neural network
# created by Bill

from tensorflow import keras as kr
import numpy as np
import matplotlib.pyplot as plt

# generate data and add random noise
x = np.linspace(-1, 1, 1000)
size = np.size(x)
np.random.shuffle(x)
y = np.cos(5 * x) + np.random.normal(0, 0.02, (size, ))

# comment to hide plot of data
# plt.plot(x, y, '^b', markersize = 4.0)
# plt.show()

# training parameters
ratio_train = 0.8 # ratio of training set, the rest are testing set
times = 1000 # training times
check_interval = 100
loss = 'mse'
optimizer = 'adam'

# layer properties
units_layer1 = 100
activation_layer1 = 'tanh'
units_layer2 = 100
activation_layer2 = 'tanh'

size_train = int(size * ratio_train) # size of training set
size_test = size - size_train # size of training set
x_train, y_train = x[: size_train], y[: size_train] # training set
x_test, y_test = x[size_train :], y[size_train :] # testing set

# initialize model
model = kr.Sequential()

model.add(kr.layers.Dense(units = units_layer1, input_dim = 1)) # add 1st hidden layer (also input layer) 
model.add(kr.layers.Activation(activation_layer1)) # set activation function of 1st hidden layer

model.add(kr.layers.Dense(units = units_layer2)) # add 2nd hidden layer
model.add(kr.layers.Activation(activation_layer2)) # set activation function of 2nd hidden layer

model.add(kr.layers.Dense(units = 1)) # add output layer
# model.add(kr.layers.Activation('tanh')) # set activation function of output layer
          
model.compile(loss = loss, optimizer = optimizer) # set loss function and optimizer

# training
print('Begin Training')
# comment to hide training method 1
for step in range(1, times + 1):
    cost = model.train_on_batch(x_train, y_train)
    if step % check_interval == 0:
        print('Step:', step, '/', times, '\t', 'Loss: ', cost)

# comment to hide training method 2
# model.fit(x_train, y_train, epochs = times, batch_size = size_train)

print('End Training\n')

# testing
print('Begin Testing')
cost = model.evaluate(x_test, y_test, batch_size = size_test)
print('Loss:', cost)
print('End Testing')

# comment to hide plot of fitting result
x = np.sort(x)
y_predict = model.predict(x)
plt.plot(x_test, y_test, '^b', markersize = 4.0) # testing data set is shown in blue scatters
plt.plot(x, y_predict, 'r') # fitting line is shown in red solid line
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.show()
