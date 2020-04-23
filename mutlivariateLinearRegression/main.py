import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from computeCost import computeCost
from gradientDescent import gradientDescent

data = pd.read_csv('data\\ex1data2.txt', sep = ',', header = None)
X = data.iloc[:,0:2] # read first two columns into X
y = data.iloc[:,2] # read the third column into y
m = len(y) # no. of training samples

X = (X - np.mean(X))/np.std(X) # feature normalization

ones = np.ones((m,1))
X = np.hstack((ones, X))
alpha = 0.01
iterations = 400
theta = np.zeros((3,1))
y = y[:,np.newaxis] 

#print(computeCost(X, y, theta))
theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)
#print(computeCost(X, y, theta))

# Plot the data
plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta), color="black")
plt.show()