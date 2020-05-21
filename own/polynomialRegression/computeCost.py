def computeCost(x, y, theta):

    m = len(y)
    error=0
    for i in range(m):
        error+=(theta[0]+theta[1]*x[i]+theta[2]*x[2]-y[i])**2

    return (1/(2*m))*error