from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

print()


#gnb = GaussianNB()

#fit = gnb.fit(X_train, y_train)
#y_pred = fit.predict(X_test)