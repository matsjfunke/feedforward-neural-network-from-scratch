"""
matsjfunke
"""
import numpy as np


def ReLU_deriv(Z):

def relu_derivative(weighted_sum):
    return np.where(weighted_sum > 0, 1, 0)


print(ReLU_deriv(3))
print(relu_derivative(3))
