import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Plot line with slope and intercept (don't know why matplotlib doesn't have this function)
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')

df = pd.read_csv("data.csv")
#print(df)

plt.xlabel("time")
plt.ylabel("cells")
plt.scatter(df.time, df.cells, color="red", marker="+")

x_df = df.drop("cells", axis="columns")
y_df = df.cells

reg = linear_model.LinearRegression()
reg.fit(x_df, y_df)

# r2 score
print(reg.score(x_df, y_df))

abline(reg.coef_, reg.intercept_)

plt.show()