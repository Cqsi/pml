import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df.Sex[df.Sex == "male"] = 1
df.Sex[df.Sex == "female"] = 0
df_test.Sex[df_test.Sex == "male"] = 1
df_test.Sex[df_test.Sex == "female"] = 0

# frick this
df.Age = df.Age.fillna(30)
df_test.Age = df_test.Age.fillna(30)

columns_drop = ["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "PassengerId"]

y_train = df.Survived
X_train = df.drop(columns_drop, axis="columns").drop("Survived", axis="columns")

gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)

X_test = df_test.drop(columns_drop, axis="columns")
y_pred = fit.predict(X_test)

# PREDICTION IS READY


passenger_ids = df_test.PassengerId.values

pd.DataFrame(np.column_stack((passenger_ids, y_pred))).to_csv("file.csv", index=None)


# TESTING WITH OWN VALUES
print(fit.predict([np.array([1, 1, 48])]))