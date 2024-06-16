"""
matsjfunke

credit https://youtu.be/w8yWXqWQYmU?si=2DmUAwDRi9NTRz7O
"""
import zipfile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load the dataset
zip_file_path = './input/train.csv.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract('train.csv', path="./input")

data = pd.read_csv("./input/train.csv")
data = np.array(data)
m, n = data.shape


# data preparation
np.random.shuffle(data)  # Shuffle the dataset

# Splitting into development (dev) and training sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]  # Labels for dev set
X_dev = data_dev[1:n]  # Features for dev set
X_dev = X_dev / 255.  # Normalize the features (scaling)

data_train = data[1000:m].T
Y_train = data_train[0]  # Labels for training set
X_train = data_train[1:n]  # Features for training set
X_train = X_train / 255.  # Normalize the features (scaling)
_, m_train = X_train.shape  # Number of training examples


# Initialize parameters for the neural network
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# activation functions
def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


# Forward propagation in the neural network
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# Derivative of ReLU activation function -> compute gradient
def ReLU_deriv(Z):
    return Z > 0


# Convert labels to one-hot encoded vectors
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# Backward propagation to compute gradients
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


# Update parameters using gradient descent
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# Get predictions based on output activations
# extracts index of highest probability class for each input example, converting softmax outputs into discrete class predictions.
def get_predictions(A2):
    return np.argmax(A2, 0)


# Calculate accuracy of predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# Gradient descent to train the neural network
# alpha = learning rate
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("accuracy", get_accuracy(predictions, Y))
    return W1, b1, W2, b2


# Train the neural network using gradient descent
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)


# Make predictions on the development set
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


# Test predictions on specific examples from the training set
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Label: ", label, "Prediction: ", prediction)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.title(f"Label: {label} Prediction: {prediction}")
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# Test some predictions on training set examples
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

# Evaluate accuracy on the development set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, Y_dev))
