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
def inital_parameters(num_input_neurons, num_output_neurons):
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


# Initialize parameters
num_input_neurons = features_train.shape[0]
num_output_neurons = 10  # Assuming 10 classes for classification

weights, biases = inital_parameters(num_input_neurons, num_output_neurons)
activation_function = "softmax"  # Example with softmax activation

# Perform forward propagation
output = forward_propagation(features_train, weights, biases, activation_function)

# Print and verify output
print("Output shape:", output.shape)  # (10, 41000): 10 classes across 41,000 training examples.
print("Output example:", output[:, 0])  # array that represents softmax probabilities (range 0-1) for one example across 10 classes.


# 5. calculate cost

# def gradient_descent(features_train, labels_train, learning_rate, max_iterations):
