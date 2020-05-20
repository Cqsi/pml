from computeCost import computeCost
from sigmoid import sigmoid

def gradient(x, y, theta, alpha, iterations):

    m = len(y)
    
    for j in range(iterations):
        error0 = 0           
        error1 = 0
        for i in range(m):
            error0+=(sigmoid(theta[0]+theta[1]*x[i])-y[i])*x[i]
            error1+=(sigmoid(theta[0]+theta[1]*x[i])-y[i])*y[i]

        temp0 = theta[0] - alpha*error0
        temp1 = theta[1] - alpha*error1

        #print(computeCost(x, y, theta))
        theta = [temp0, temp1]

    return theta