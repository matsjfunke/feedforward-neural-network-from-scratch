"""
matsjfunke
"""

import argparse
import zipfile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

dashline = "-" * 100

# 1. dataset loading
zip_file_path = "./input/train.csv.zip"
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extract("train.csv", path="./input")

data = pd.read_csv("./input/train.csv")
data = np.array(data)
row, col = data.shape

# 2. dataset preparation
np.random.shuffle(data)  # Shuffle the dataset

# Splitting data into train and test sets
data_test = data[0:1000].T
labels_test = data_test[0]
features_test = data_test[1:col]
# normalizing features_test ensures numbers are between 0-1
features_test = features_test / 255.0  # pixel value can range from 0 to 255

data_train = data[1000:row].T
labels_train = data_train[0]
features_train = data_train[1:col]
features_train = features_train / 255.0


# 3. generate random starting weights & biases
def initial_parameters(layers):
    parameters = {}
    num_layers = len(layers)

    for layer in range(1, num_layers):
        parameters[f"W{layer}"] = np.random.rand(layers[layer], layers[layer - 1]) - 0.5
        parameters[f"b{layer}"] = np.random.rand(layers[layer], 1) - 0.5

    return parameters


# activation functions -> input: weighted_sum of inputs * weights + bias
def relu(weighted_sum):
    return np.maximum(weighted_sum, 0)


def softmax(weighted_sum):
    exp_sum = np.exp(weighted_sum - np.max(weighted_sum, axis=0, keepdims=True))
    return exp_sum / exp_sum.sum(axis=0, keepdims=True)


# 4. forward propagation
def forward_prop(features, parameters, activation_functions):
    cache = {}
    activations = features
    cache["A0"] = activations  # Input features are treated as A0
    num_layers = len(parameters) // 2

    # Initialize cache['Z0'] for the weighted input to the first layer
    cache["Z0"] = features

    for layer in range(1, num_layers + 1):
        weights = parameters[f"W{layer}"]
        biases = parameters[f"b{layer}"]
        weighted_sum = weights.dot(activations) + biases
        cache[f"Z{layer}"] = weighted_sum

        if activation_functions[layer - 1] == "relu":
            activations = relu(weighted_sum)
        elif activation_functions[layer - 1] == "softmax":
            activations = softmax(weighted_sum)

        cache[f"A{layer}"] = activations

    return activations, cache


# 5. cost function calculation
def cost_function(predictions, labels):
    sample_count = labels.shape[1]
    epsilon = 1e-10  # Small epsilon to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)  # Clip predictions to [epsilon, 1-epsilon]
    cost = -np.sum(labels * np.log(predictions)) / sample_count
    print("Cost: ", cost)
    return cost


# one-hot encode labels_train
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((num_classes, labels.size))
    one_hot[labels, np.arange(labels.size)] = 1
    return one_hot


# derivatices of activation functions
def relu_derivative(weighted_sum):
    return np.where(weighted_sum > 0, 1, 0)


# 6. backward propagation to compute gradients
def backward_prop(features, labels, parameters, cache, activation_functions):
    gradients = {}
    num_layers = len(parameters) // 2
    sample_count = features.shape[1]
    one_hot_labels = one_hot_encode(labels, parameters[f"W{num_layers}"].shape[0])

    # Initialize error for the last layer
    error = cache[f"A{num_layers}"] - one_hot_labels

    # Backpropagate through each layer
    for layer in reversed(range(1, num_layers + 1)):
        delta = error

        if activation_functions[layer - 1] == "softmax":
            # Calculate gradients for softmax activation
            # softmax_derivative not used because softmax used in output layer where we calculate the cross-entropy loss directly
            gradient_weights = 1.0 / sample_count * delta.dot(cache[f"A{layer-1}"].T)
            gradient_biases = 1.0 / sample_count * np.sum(delta, axis=1, keepdims=True)
            # Calculate error for previous layer
            error = parameters[f"W{layer}"].T.dot(delta)
        elif activation_functions[layer - 1] == "relu":
            # Calculate gradients for ReLU activation
            gradient_weights = 1.0 / sample_count * delta.dot(cache[f"A{layer-1}"].T)
            gradient_biases = 1.0 / sample_count * np.sum(delta, axis=1, keepdims=True)
            # Calculate error for previous layer using ReLU derivative
            error = parameters[f"W{layer}"].T.dot(delta) * relu_derivative(cache[f"Z{layer-1}"])

        # Store gradients for weights and biases
        gradients[f"dW{layer}"] = gradient_weights
        gradients[f"db{layer}"] = gradient_biases

    return gradients


# 7. update parameters (uses gradient_descent formula)
def gradient_descent(parameters, gradients, learning_rate):
    num_layers = len(parameters) // 2

    for layer in range(1, num_layers + 1):
        parameters[f"W{layer}"] -= learning_rate * gradients[f"dW{layer}"]
        parameters[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"]

    return parameters


# 8. train network
def train_model(features_train, labels_train, layers, max_iterations, learning_rate, tolerance):
    num_input_neurons = features_train.shape[0]
    layers = [num_input_neurons] + layers
    activation_functions = ["relu"] * (len(layers) - 2) + ["softmax"]

    parameters = initial_parameters(layers)

    previous_cost = float("inf")

    for iteration in range(max_iterations):
        layer_output, cache = forward_prop(features_train, parameters, activation_functions)
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


# 9. define architecture & start training network
def run_architecture(features_train, labels_train, layers, activation_functions, max_iterations=100, learning_rate=0.04, tolerance=0.002):
    parameters = train_model(features_train, labels_train, layers, max_iterations, learning_rate, tolerance)
    for i in range(1, len(layers)):
        print(f"Layer {i} - Weights shape: {parameters[f'W{i}'].shape}, Biases shape: {parameters[f'b{i}'].shape}")

    return parameters


# 10. predict function
def predict(features, parameters, activation_functions):
    predictions, _ = forward_prop(features, parameters, activation_functions)
    return np.argmax(predictions, axis=0)


# 11. use parameters to predict a specific sample
def predict_single_example(sample_index, features_test, parameters, activation_functions):
    sample = features_test[:, sample_index, None]
    features = sample.reshape((-1, 1))  # Ensure it's a column vector

    predictions, _ = forward_prop(features, parameters, activation_functions)
    return np.argmax(predictions)


# visualize input number / pixels & compare to model prediction
def plot_sample_prediction(index, features_test, labels_test, predicted_label):
    current_image = features_test[:, index, None]
    label = labels_test[index]
    current_image = current_image.reshape((28, 28)) * 255

    plt.gray()
    plt.title(f"Labeled as: {label}, predicted as: {predicted_label}")
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural network architecture.")
    parser.add_argument("--layers", type=int, nargs="+", default=[128, 64, 10], help="Number of neurons in each layer")
    parser.add_argument("--activations", type=str, nargs="+", default=["relu", "relu", "softmax"], help="Activation functions for each layer")

    args = parser.parse_args()
    layers = args.layers
    activation_functions = args.activations

    # Assuming features_train, labels_train, features_test, labels_test are already loaded
    parameters = run_architecture(features_train, labels_train, layers, activation_functions)

    predictions_test = predict(features_test, parameters, activation_functions)
    accuracy = accuracy_score(labels_test, predictions_test)
    dashline = "-" * 40
    print(f"{dashline}\nAccuracy on test set: {accuracy * 100:.2f}%\n{dashline}")

    predicted_label = predict_single_example(0, features_test, parameters, activation_functions)
    print("Testing trained parameters on first sample of test dataset")
    print(f"Labeled as: {labels_test[0]}, predicted as: {predicted_label}\n{dashline}")
    plot_sample_prediction(0, features_test, labels_test, predicted_label)

    predicted_label = predict_single_example(1, features_test, parameters, activation_functions)
    print("Testing trained parameters on second sample of test dataset")
    print(f"Labeled as: {labels_test[1]}, predicted as: {predicted_label}\n{dashline}")
    plot_sample_prediction(1, features_test, labels_test, predicted_label)
