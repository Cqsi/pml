import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Split train/test
from sklearn.model_selection import train_test_split

# matplotlib function
from line import abline

df = pd.read_csv("data.csv")
#print(df)

plt.xlabel("time")
plt.ylabel("cells")
plt.scatter(df.time, df.cells, color="red", marker="+")

x_df = df.drop("cells", axis="columns")
y_df = df.cells

# Split the data set into train/test, test_size in percent
# X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.4, random_state=10)

reg = linear_model.LinearRegression()
reg.fit(x_df, y_df)

# r2 score
# print(reg.score(x_df, y_df))

# print("Mean squared error between y_test and predicted =", np.mean(prediction_test-y_test)**2)

abline(reg.coef_, reg.intercept_)
plt.show()