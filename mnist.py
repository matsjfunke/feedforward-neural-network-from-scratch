"""
matsjfunke
"""
import os
import math
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# The labeled training dataset consists of 42000 images, each of size 28x28 = 784 pixels. Labels are from 0 to 9 for pixelbrightness
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# separating labels and pixels
train_labels = np.array(train_data.loc[:, 'label'])
train_data = np.array(train_data.loc[:, train_data.columns != 'label'])


# input number visualization.
index = 7  # change  index to view other numbers
plt.title((train_labels[index]))
plt.imshow(train_data[index].reshape(28, 28), cmap="binary")
plt.show()


# distribution of numbers 0-9 accross training set
print("train data")
y_value = np.zeros((1, 10))
for i in range(10):
    print("occurance of ", i, "=", np.count_nonzero(train_labels == i))
    y_value[0, i-1] = np.count_nonzero(train_labels == i)

y_value = y_value.ravel()
x_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.xlabel('number label')
plt.ylabel('count')
plt.bar(x_value, y_value, 0.7, color='r')
plt.xticks(x_value)
plt.show()


# ensuring that train_data is a vector, and train_label is a matrix of 0's and 1's
train_data = np.reshape(train_data, [784, 42000])  # reshapes the training data, each image: originally 28x28 pixels is flattened into vector of 784 elements
train_label = np.zeros((10, 42000))  # one-hot encoding transforms categorical labels into  binary format
for col in range(42000):
    val = train_labels[col]
    for row in range(10):
        if (val == row):
            train_label[val, col] = 1
print("train_data shape=" + str(np.shape(train_data)))
print("train_label shape=" + str(np.shape(train_label)))


# activation functions (forward propagation)
# Z: linear transformation (pre-activation value)
def sigmoid(Z):
    activation_function = 1 / (1 + np.exp(-Z))
    cache = Z
    return activation_function, cache


def relu(Z):
    activation_function = np.maximum(0, Z)
    cache = Z
    return activation_function, cache


# derivative of activation function (backward propagation)
# remember gradients are used to find the minimum
def sigmoid_gradient(activation_fn_gradient, cache):
    Z = cache
    sigmoid_Z = 1 / (1 + np.exp(-Z))
    linear_transformation_gradient = activation_fn_gradient * sigmoid_Z * (1 - sigmoid_Z)

    assert (linear_transformation_gradient.shape == Z.shape)
    return linear_transformation_gradient


def relu_gradient(activation_fn_gradient, cache):
    Z = cache
    linear_transformation_gradient = np.array(activation_fn_gradient, copy=True)
    linear_transformation_gradient[Z <= 0] = 0

    assert (linear_transformation_gradient.shape == Z.shape)
    return linear_transformation_gradient
