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
print(f"dataset contains {row} samples and each sample has {col - 1}pixels and 1 label")

np.random.shuffle(data)  # Shuffle the dataset

data_train = data[1000:row].T
labels = data_train[0]  # Labels for training set
features = data_train[1:col]
features = features / 255.  # Normalize the features (scaling)


def visualize_data_distribution():
    print(dashline, "\ndistribution of labels / numbers")
    label_column = data[:, 0]
    unique_labels, counts = np.unique(label_column, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label: {label}, Count: {count}")
    print(dashline)

    # plot Distribution
    plt.xlabel('Labels / Numbers')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Labels')
    plt.xticks(unique_labels)
    plt.bar(unique_labels, counts, 0.7, color='r')
    plt.show()


# visualize input number / pixels
def plot_pixels(index):
    print("Ploting the pixels to visualize number, predicting the label using trained parameters")

    current_image = features[:, index, None]
    label = labels[index]
    print(f"plot of of number labeled as {label}")

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.title(f"Labeled as: {label}")
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    visualize_data_distribution()
    plot_pixels(1)
    plot_pixels(3)
