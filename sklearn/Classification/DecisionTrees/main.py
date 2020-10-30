from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split

import numpy as np


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = tree.DecisionTreeClassifier()

fit = clf.fit(X_train, y_train)
y_pred = fit.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))


# Test the prediction with own values below, e.g go to 
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# and take some values from there

pred_test = np.array([6.4, 3.2,	5.3, 2.3])

pred = fit.predict([pred_test])
print(pred[0])