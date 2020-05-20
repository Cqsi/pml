import math
from sigmoid import sigmoid

def computeCost(x, y, theta):

    m = len(y)
    error=0
    for i in range(m):
        error+=(-y[i]*math.log(sigmoid(theta[0]+theta[1]*x[i]))+(y[i]-1)*(math.log(1-sigmoid(theta[0]+theta[1]*x[i]))))

    return (1/m)*error