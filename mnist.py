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
print(len(train_labels))
print(len(train_data))


# Visualize the input data. Change the index value to visualize the particular index data.
index = 7
plt.title((train_labels[index]))
plt.imshow(train_data[index].reshape(28, 28), cmap="binary")
plt.show()
