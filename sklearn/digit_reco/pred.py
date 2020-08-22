import matplotlib.pyplot as plt

def test(a, X_test, predicted):
    fig = plt.figure()
    arr = [[], [], [], [], [], [], [], []]

    for i in range(len(X_test[a])):
        arr[int(i/8)].append(X_test[a][i])

    plt.imshow(arr, cmap = plt.cm.gray_r)
    plt.show()
    print(predicted[a])