import numpy as np

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
