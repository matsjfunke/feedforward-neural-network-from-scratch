"""
matsjfunke
"""
import zipfile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
dashline = "-" * 100

# 1. dataset loading
zip_file_path = './input/train.csv.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract('train.csv', path="./input")

data = pd.read_csv("./input/train.csv")
data = np.array(data)
row, col = data.shape

# visualize data
print(dashline, "\ndistribution of labels / numbers")
label_column = data[:, 0]
unique_labels, counts = np.unique(label_column, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Label: {label}, Count: {count}")

plt.xlabel('Labels / Numbers')
plt.ylabel('Number of Samples')
plt.title('Distribution of Labels')
plt.xticks(unique_labels)
plt.bar(unique_labels, counts, 0.7, color='r')
plt.show()

# 2. dataset preparation
np.random.shuffle(data)  # Shuffle the dataset

# Splitting data into train and test sets
data_test = data[0:1000].T
labels_test = data_test[0]
features_test = data_test[1:col]
# normalizing features_test ensures numbers are between 0-1
features_test = features_test / 255.  # pixel value can range from 0 to 255

data_train = data[1000:row].T
labels_train = data_train[0]
features_train = data_train[1:col]
features_train = features_train / 255.
row_train, col_train = features_train.shape


# 3. generate random starting weights & biases
def initial_parameters(num_input_neurons, num_output_neurons):
    weights = np.random.rand(num_output_neurons, num_input_neurons) - 0.5
    biases = np.random.rand(num_output_neurons, 1) - 0.5  # 1 because each layer has it's own bias
    return weights, biases


# activation functions -> input: weighted_sum of inputs * weights + bias
def relu(weighted_sum):
    return np.maximum(weighted_sum, 0)


def softmax(weighted_sum):
    return np.exp(weighted_sum) / sum(np.exp(weighted_sum))


# 4. forward propagation
def forward_propagation(features_train, weights, biases, activation_function):
    weighted_sum = weights.dot(features_train) + biases
    if activation_function == "relu":
        layer_output = relu(weighted_sum)
        return layer_output
    elif activation_function == "softmax":
        layer_output = softmax(weighted_sum)
        return layer_output


# 5. cost function calculation
def cost_function(predictions, labels):
    sample_count = labels.shape[1]  # Number of samples
    cost = -np.sum(labels * np.log(predictions)) / sample_count
    print("Cost: ", cost)
    return cost


# one-hot encode labels_train
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((num_classes, labels.size))
    one_hot[labels, np.arange(labels.size)] = 1
    return one_hot


def relu_derivative(weighted_sum):
    return np.where(weighted_sum > 0, 1, 0)


def softmax_derivative(softmax_output):
    s = softmax_output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


# 6. backward propagation to compute gradients
def backward_prop(features, labels, weights, layer_output, activation_function):
    sample_count = features.shape[1]  # Number of samples
    one_hot_Y = one_hot_encode(labels, weights.shape[0])  # Number of output neurons

    if activation_function == "softmax":
        output_error = layer_output - one_hot_Y  # Difference between prediction and actual output influences parameter changes
    elif activation_function == "relu":
        output_error = (layer_output - one_hot_Y) * relu_derivative(layer_output)  # Adjust parameters based on error magnitude and direction

    gradient_weights = 1 / sample_count * output_error.dot(features.T)
    gradient_biases = 1 / sample_count * np.sum(output_error, axis=1, keepdims=True)

    return gradient_weights, gradient_biases


# 7. update parameters (uses gradient_descent formula)
def gradient_descent(weights, biases, gradient_weights, gradient_biases, learning_rate):
    updated_weights = weights - learning_rate * gradient_weights
    updated_biases = biases - learning_rate * gradient_biases
    return updated_weights, updated_biases


# Example gradient_descen testing loop
num_input_neurons = features_train.shape[0]
num_output_neurons = len(np.unique(labels_train))
learning_rate = 0.01
epochs = 100

# Initialize parameters
weights, biases = initial_parameters(num_input_neurons, num_output_neurons)

print(dashline)
for epoch in range(epochs):
    # Forward propagation
    layer_output = forward_propagation(features_train, weights, biases, activation_function="softmax")

    # Compute cost
    current_cost = cost_function(layer_output, one_hot_encode(labels_train, num_output_neurons))

    # Backward propagation
    gradient_weights, gradient_biases = backward_prop(features_train, labels_train, weights, layer_output, activation_function="softmax")

    # Update parameters using gradient descent
    updated_weights, updated_biases = gradient_descent(weights, biases, gradient_weights, gradient_biases, learning_rate)

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Cost: {current_cost}")
        print(f"original weights: {weights.shape} updated: {updated_weights.shape}")
        print(f"original biases: {biases.shape} updated: {updated_biases.shape}")

# After training, you can evaluate on test set or visualize results
# For example:
test_predictions = forward_propagation(features_test, weights, biases, activation_function="softmax")
test_cost = cost_function(test_predictions, one_hot_encode(labels_test, num_output_neurons))
print(f"Test Cost: {test_cost}")
