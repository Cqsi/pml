from loadData import loadData
from computeCost import computeCost
from gradientDescent import gradient
from plot import plot

import pandas as pd

path = "C:\\Users\\Petter\\Desktop\\PythonProjects\\machineLearning\\own\\logisticRegression\\data.txt"

data = loadData(path, None)

theta = [0,0]
alpha = 0.01
iterations = 1500

plot(data)
