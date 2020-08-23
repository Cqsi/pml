import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def test(a, X_test, predicted):
    fig = plt.figure()
    arr = [[], [], [], [], [], [], [], []]

    for i in range(len(X_test[a])):
        arr[int(i/8)].append(X_test[a][i])

    plt.imshow(arr, cmap = plt.cm.gray_r)
    plt.show()
    print(predicted[a])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# add own image
def add_digit(path):
    img = mpimg.imread(path)
    gray = rgb2gray(img)
    a=(16-gray*16).astype(int) # really weird here, but try to convert to 0..16
    plt.imshow(a, cmap = plt.get_cmap('gray_r'))
    plt.show()
    
    return a
