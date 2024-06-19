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
print(dashline)

plt.xlabel('Labels / Numbers')
plt.ylabel('Number of Samples')
plt.title('Distribution of Labels')
plt.xticks(unique_labels)
plt.bar(unique_labels, counts, 0.7, color='r')
plt.show()
