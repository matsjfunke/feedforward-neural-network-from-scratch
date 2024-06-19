import numpy as np
import pandas as pd

# Read and preprocess data
data = pd.read_csv("./input/train.csv")
data = np.array(data)
row, col = data.shape

# Shuffle the dataset
np.random.shuffle(data)

# Splitting data into train and test sets
data_test = data[0:1000].T
labels_test = data_test[0]
features_test = data_test[1:col]
features_test = features_test / 255.

data_train = data[1000:row].T
labels_train = data_train[0]
features_train = data_train[1:col]
features_train = features_train / 255.

# Define functions for initial parameters, activations, and forward propagation
def initial_parameters(layers):
    parameters = {}
    num_layers = len(layers)
    
    for l in range(1, num_layers):
        parameters[f'W{l}'] = np.random.rand(layers[l], layers[l-1]) - 0.5
        parameters[f'b{l}'] = np.random.rand(layers[l], 1) - 0.5
        
    return parameters

def relu(weighted_sum):
    return np.maximum(weighted_sum, 0)

def softmax(weighted_sum):
    exp_sum = np.exp(weighted_sum - np.max(weighted_sum, axis=0))
    return exp_sum / exp_sum.sum(axis=0)

def forward_propagation(features, parameters, activation_functions):
    cache = {}
    A = features
    cache['A0'] = A  # Input features are treated as A0
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = W.dot(A) + b
        cache[f'Z{l}'] = Z
        
        if activation_functions[l-1] == "relu":
            A = relu(Z)
        elif activation_functions[l-1] == "softmax":
            A = softmax(Z)
        
        cache[f'A{l}'] = A
        
    return A, cache

# Define functions for cost calculation, one-hot encoding, backward propagation, and gradient descent
def cost_function(predictions, labels):
    sample_count = labels.shape[1]
    cost = -np.sum(labels * np.log(predictions)) / sample_count
    print("Cost: ", cost)
    return cost

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((num_classes, labels.size))
    one_hot[labels, np.arange(labels.size)] = 1
    return one_hot

def relu_derivative(weighted_sum):
    return np.where(weighted_sum > 0, 1, 0)

def backward_prop(features, labels, parameters, cache, activation_functions):
    gradients = {}
    L = len(parameters) // 2
    m = features.shape[1]
    one_hot_Y = one_hot_encode(labels, parameters[f'W{L}'].shape[0])
    
    dA = cache[f'A{L}'] - one_hot_Y
    
    for l in reversed(range(1, L + 1)):
        dZ = dA
        dW = 1/m * dZ.dot(cache[f'A{l-1}'].T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA = parameters[f'W{l}'].T.dot(dZ) * relu_derivative(cache[f'Z{l-1}'])
        
        gradients[f'dW{l}'] = dW
        gradients[f'db{l}'] = db
    
    return gradients

def gradient_descent(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    
    return parameters

# Define the training function
def train_layer(features_train, labels_train, layers, max_iterations, learning_rate, tolerance):
    num_input_neurons = features_train.shape[0]
    layers = [num_input_neurons] + layers
    activation_functions = ["relu"] * (len(layers) - 2) + ["softmax"]
    
    parameters = initial_parameters(layers)
    
    previous_cost = float('inf')
    
    for iteration in range(max_iterations):
        layer_output, cache = forward_propagation(features_train, parameters, activation_functions)
        current_cost = cost_function(layer_output, one_hot_encode(labels_train, layers[-1]))
        
        if abs(previous_cost - current_cost) < tolerance:
            print(f"The difference between costs was less than the tolerance: {tolerance} at iteration: {iteration + 1} with cost {current_cost}")
            break
        
        gradients = backward_prop(features_train, labels_train, parameters, cache, activation_functions)
        parameters = gradient_descent(parameters, gradients, learning_rate)
        
        previous_cost = current_cost
        
        if (iteration + 1) % 10 == 0:
            print(f"Epoch {iteration + 1}/{max_iterations}, Cost: {current_cost}")
    
    return parameters

# Run the architecture
def run_architecture(features_train, labels_train):
    layers = [128, 64, 10]
    max_iterations = 20
    learning_rate = 0.2
    tolerance = 1e-6
    
    parameters = train_layer(features_train, labels_train, layers, max_iterations, learning_rate, tolerance)
    for i in range(1, len(layers)):
        print(f"Layer {i} - Weights shape: {parameters[f'W{i}'].shape}, Biases shape: {parameters[f'b{i}'].shape}")

run_architecture(features_train, labels_train)
