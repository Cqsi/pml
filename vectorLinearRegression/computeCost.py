import numpy as np

def computeCost(X, y, theta, m):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)