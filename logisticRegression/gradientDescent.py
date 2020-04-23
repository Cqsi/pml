import numpy as np

from sigmoid import sigmoid

def gradientDescent(theta, X, y):
    m = len(y)
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))