from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import pandas as pd

df = pd.read_csv("Iris.csv")
y = df.Species
y[y == "Iris-setosa"] = 1
y[y == "Iris-versicolor"] = 2
y[y == "Iris-virginica"] = 3


#df["Species"][df["Species"] == ""]

