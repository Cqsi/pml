import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from computeCost import computeCost

data = pd.read_csv('data\\ex1data1.txt', header = None) #read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
#print(data.head()) - View first few rows of the data

# Plot the data
#plt.scatter(X, y)
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $10,000s')
#plt.show()

X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term

#print(computeCost(X, y, theta, m))