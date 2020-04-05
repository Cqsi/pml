import matplotlib.pyplot as plt
import numpy as np

def plot(x, y, theta):

    plt.scatter(x, y, label="skitscat", color="k")
    
    x = np.array(x)
    plt.plot(x, x*theta[1]+theta[0], "r")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Result")
    plt.legend()
    plt.grid()
    plt.show()