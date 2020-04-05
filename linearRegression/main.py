from loadData import loadData
from computeCost import computeCost
from gradientDescent import gradient

path = "machineLearning\\linearRegression\\data.txt"

x = loadData(path)[0]
y = loadData(path)[1]

theta = [0,0]
alpha = 0.1
iterations = 1500

print(gradient(x, y, theta, alpha, iterations))