import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

def plot(data):

    X = data.iloc[:, :-1]
    print(X)

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]
    print(y)

    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]

    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]

    # plots
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    plt.legend()
    plt.show()