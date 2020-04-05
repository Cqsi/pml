from loadData import loadData
from computeCost import computeCost

path = "linearRegression\data.txt"

x = loadData(path)[0]
y = loadData(path)[1]

theta = [0, 1]
print(computeCost(x, y, theta))