import numpy as np
import matplotlib.pyplot as plt

# Plot line with slope and intercept (don't know why matplotlib doesn't have this function)
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')