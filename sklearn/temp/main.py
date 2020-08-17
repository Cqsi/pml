from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", version=1)
#print(mnist.keys())

X, y = mnist["data"], mnist["target"]

# display an image from the dataset
digit_image = X[0].reshape(28,28)

# label of the digit above
# print(y[0])

y = y.astype(np.uint8)

plt.imshow(digit_image, cmap="binary")
plt.axis("off")
plt.show()