import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# quick program, this is how you display a digit from the digits dataset

digits = datasets.load_digits()

fig = plt.figure()
plt.imshow(digits.images[23], cmap = plt.cm.gray_r)
plt.show()

print(digits.images[23])