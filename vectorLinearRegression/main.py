import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from computeCost import computeCost
from gradientDescent import gradientDescent
from plot import plot

data = pd.read_csv('data\\ex1data1.txt', header = None) #read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
#print(data.head()) - View first few rows of the data

# temporary solution for plot function idk lol
tempX = X
tempY = y

X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term

#print(computeCost(X, y, theta))
theta = gradientDescent(X, y, theta, alpha, iterations)

plot(tempX, tempY, theta)
