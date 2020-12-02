from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

X_full = pd.read_csv("data\\train.csv", index_col="Id")
X_test_full = pd.read_csv("data\\test.csv", index_col="Id")

y = X_full["SalePrice"]
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

print(X.head())