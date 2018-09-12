# 'Keras_linear_fitting' is used to fit linear data with one layer
# created by Bill

from tensorflow import keras as kr
import numpy as np
import matplotlib.pyplot as plt

# generate data and add random noise
x = np.linspace(-1, 1, 200)
np.random.shuffle(x)
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (200, ))

# uncomment to show data
# plt.plot(x, y, '^b', markersize = 4.0)
# plt.show()

x_train, y_train = x[: 160], y[: 160] # training set
x_test, y_test = x[160 :], y[160 :] # testing set

# initialize model
model = kr.Sequential()
model.add(kr.layers.Dense(units = 1, input_dim = 1)) # add input layer (also output layer)
model.compile(loss = 'mse', optimizer = 'sgd') # set loss function and optimizer

# training
print('Begin Training')
for step in range(301): # repeat 301 times
    cost = model.train_on_batch(x_train, y_train)
    if step % 100 == 0:
        print('Loss: ', cost)
print('End Training\n')

# testing
print('Begin Testing')
cost = model.evaluate(x_test, y_test, batch_size = 40)
print('Loss:', cost)
print('End Testing')

# print weight and bias
W, b = model.layers[0].get_weights()
print('Weight=', W, '\nBias=', b)

# uncomment to show fitting result
# x = np.sort(x)
# y_predict = model.predict(x)
# plt.plot(x_test, y_test, '^b', markersize = 4.0)
# plt.plot(x, y_predict, 'r')
# plt.axis([-1.1, 1.1, -1.1, 1.1])
# plt.show()