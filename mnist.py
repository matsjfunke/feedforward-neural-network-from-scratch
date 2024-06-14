"""
matsjfunke
"""
import os
import math
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# 1. data loading
# The labeled training dataset consists of 42000 images, each of size 28x28 = 784 pixels. Labels are from 0 to 9 for pixelbrightness
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# separating labels and pixels
train_labels = np.array(train_data.loc[:, 'label'])
train_data = np.array(train_data.loc[:, train_data.columns != 'label'])


# 2. data visualization
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


# 3. data preparation
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


# 3. forward propagation
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


def softmax(Z):
    exponential_Z = np.exp(Z)
    activation_function = exponential_Z / np.sum(exponential_Z)
    cache = Z
    return activation_function, cache


# initializing the parameters (weights and biases)
def initialize_parameters_deep(layer_dims):
    parameters = {}
    layer_count = len(layer_dims)

    for layer in range(1, layer_count):
        parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) / np.sqrt(layer_dims[layer - 1])  # *0.01
        parameters['b' + str(layer)] = np.zeros((layer_dims[layer], 1))

    return parameters


# Computes the linear part of a layer's forward propagation
# A: activations from previous layer (or input data)
# W: weight matrix of current layer
# b: bias vector of current layer
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    assert (Z.shape == (W.shape[0], A.shape[1]))
    return Z, cache


# Computes forward propagation layer with activation functions
# A_prev: activations from previous layer (or input data)
# W: weight matrix of current layer
# b: bias vector of current layer
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        # print("Z="+str(Z))
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)

    return A, cache


# Computes forward propagation for the entire neural network
# X: input data
# parameters: dictionary containing parameters (weights and biases) of the model
def L_model_forward(X, parameters):
    caches = []
    A = X
    network_layer = len(parameters) // 2

    # Forward propagation for layers 1 to (L-1) with ReLU activation
    for layer in range(1, network_layer):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(layer)], parameters['b' + str(layer)], activation="relu")
        caches.append(cache)

    # Forward propagation for the last layer (L) with softmax activation
    AL, cache = linear_activation_forward(A, parameters['W' + str(network_layer)], parameters['b' + str(network_layer)], activation="softmax")
    caches.append(cache)
    return AL, caches


# 4. cost calculation


# 5. backward propagation
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
