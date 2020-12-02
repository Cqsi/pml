from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

le = LabelEncoder()

df = pd.read_csv("Iris.csv")
y = df.Species
X = df.drop("Species", axis=1).drop("Id", axis=1)

# turn strings into numeric values
le.fit(y)
y = le.transform(y)

print(y[:10])
#print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(metrics.r2_score(y_pred, y_test))

# confusion matrix
#disp = metrics.plot_confusion_matrix(gnb, X_test, y_test)
#disp.figure_.suptitle("Confusion Matrix")
#plt.show()

print("Prediction: " + le.inverse_transform(fit.predict([np.array([6, 3.05, 5.0, 1.85])]))[0])