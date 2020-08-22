import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

mnist = fetch_openml("mnist_784", version=1)
#print(mnist.keys())

X, y = mnist["data"], mnist["target"]

# display an image from the dataset
digit_image = X[0].reshape(28,28)

# label of the digit above
# print(y[0])

y = y.astype(np.uint8)

#plt.imshow(digit_image, cmap="binary")
#plt.axis("off")
#plt.show()

# split the data into train and test sets
# this could also be done using sklearn
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5) # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)

sdg_clf = SGDClassifier(random_state=42)
sdg_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([some_digit]))