import numpy as np

from sigmoid import sigmoid

def costFunction(theta, X, y):
    m = len(y)
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) + np.multiply((1-y),np.log(1-sigmoid(X @ theta))))
    return J