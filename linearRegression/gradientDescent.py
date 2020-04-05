from computeCost import computeCost

def gradient(x, y, theta, alpha, iterations):

    m = len(y)
    
    for j in range(iterations):
        error0 = 0           
        error1 = 0
        for i in range(m):
            error0+=theta[0]+theta[1]*x[i]-y[i]
            error1+=(theta[0]+theta[1]*x[i]-y[i])*x[i]

        temp0 = theta[0] - (alpha/m)*error0
        temp1 = theta[1] - (alpha/m)*error1

        print(computeCost(x, y, theta))
        theta = [temp0, temp1]

    return theta