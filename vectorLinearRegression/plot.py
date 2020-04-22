import matplotlib.pyplot as plt
import numpy as np

def plot(x, y, theta):

    plt.scatter(x, y, label="Input", color="k")
    
    for i in range(1, 11):
        x.append(x[-1]+1)
    x = np.array(x)
    plt.plot(x, x*theta[1]+theta[0], "r", label="Linear Regression")
    plt.axis("square")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Result")
    plt.legend()
    plt.grid()
    plt.show()