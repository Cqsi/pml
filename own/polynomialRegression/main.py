from loadData import loadData
from computeCost import computeCost
from gradientDescent import gradient
from plot import plot

path = "C:\\Users\\Petter\\Desktop\\PythonProjects\\machineLearning\\own\\polynomialRegression\\data.txt"

x = loadData(path)[0]
y = loadData(path)[1]

theta = [0,0,0]
alpha = 0.01
iterations = 1500

print(computeCost(x,y,theta))
theta = gradient(x, y, theta, alpha, iterations)
print(computeCost(x,y,theta))
print(theta)

plot(x,y,theta)