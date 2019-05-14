import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale, MinMaxScaler
import scipy.io as scio
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def de_sigmoid(z):
    return z * (1 - z)


def auto_encoder(x, iter=300, layer_size=100, p=0.01, alpha=1.0, lambd=0, beta=0.1):
    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    np.random.seed(5)
    low = np.sqrt(6 / (n+layer_size))
    w_1 = np.random.uniform(-low, low, (n + 1, layer_size))
    w_2 = np.random.uniform(-low, low, (layer_size + 1, n))
    hidden_layer = np.ones((m, layer_size+1))

    for i in range(iter):
        # 前向传播
        hidden_layer[:, 1:] = sigmoid(np.dot(x, w_1))
        x_hat = sigmoid(np.dot(hidden_layer, w_2))  # (m, n)

        if 0:
            epsilon = np.zeros_like(w2)
            epsilon[1, 1] = 1e-4
            x_hat = h_ @ w2
            x_hat1 = h_ @ (w2 + epsilon)
            x_hat2 = h_ @ (w2 - epsilon)
            p_hat = np.mean(h, 0)
            kl = -np.sum(p * np.log(p_hat) + (1 - p) * np.log(1 - p_hat))
            err1 = x_hat1 - x
            cost1 = (np.sum(err1.T @ err1) / 2 + beta * kl) / m
            err2 = x_hat2 - x
            cost2 = (np.sum(err2.T @ err2) / 2 + beta * kl) / m

            err = x_hat - x
            delta2 = err
            grad2 = h_.T @ delta2 / m

            print(grad2[1, 1] - ((cost1 - cost2) / (2 * epsilon[1, 1])))

        p_hat = np.mean(hidden_layer[:, 1:], 0)
        err = x_hat - x[:, 1:]
        kl = -np.sum((p * np.log(p_hat) + (1-p) * np.log(1-p_hat)))
        weight_decay = np.sum(np.dot(w_1[1:].T, w_1[1:])) + np.sum(np.dot(w_2[1:].T, w_2[1:]))
        cost = (np.sum(np.dot(err.T, err))/2 + beta * kl) / m + lambd * weight_decay / 2
        if i % 5 == 0:
            print(i, ' : ', cost)

        # 反向传播
        delta_2 = err * de_sigmoid(x_hat)  # (m, n)
        delta_w = w_2.copy()
        delta_w[0] = 0
        grad_2 = alpha * np.dot(hidden_layer.T, delta_2) / m + lambd * delta_w

        delta_1 = (np.dot(delta_2, w_2[1:].T) + beta * (-p/p_hat + (1-p)/(1-p_hat)).reshape(1, -1)) * de_sigmoid(hidden_layer[:, 1:])
        delta_w = w_1.copy()
        delta_w[0] = 0
        grad_1 = 4 * alpha * np.dot(x.T, delta_1) / m + lambd * delta_w

        w_2 -= grad_2
        w_1 -= grad_1

    return w_1, hidden_layer[:, 1:], x_hat


def display_data(imgData):
    sum = 0
    '''
    显示100个数（若是一个一个绘制将会非常慢，可以将要画的数字整理好，放到一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    '''
    m, n = imgData.shape
    m, n = int(np.sqrt(m)), int(np.sqrt(n))
    pad = 1
    display_array = -np.ones((pad + m * (n + pad), pad + m * (n + pad)))
    for i in range(m):
        for j in range(m):
            display_array[pad + i * (n + pad):pad + i * (n + pad) + n,
            pad + j * (n + pad):pad + j * (n + pad) + n] = (
                imgData[sum, :].reshape(n, n, order="F"))
            sum += 1

    plt.figure()
    plt.imshow(display_array, cmap='gray')  # 显示灰度图像
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    path = 'F:\\UFLDL\\data_digits.mat'
    data = scio.loadmat(path)
    X = data['X']
    Y = data['y']
    # X = scale(X)  # 预处理不如不处理
    # X = (X + 2) / 4

    w, h, x_hat = auto_encoder(X, layer_size=100, iter=200, p=0.02, alpha=0.5, lambd=0.0001, beta=0.01)
    w_show = w[1:, :100].T
    # diag = np.diag(w_show @ w_show.T).reshape(-1, 1)
    # w_show /= np.sqrt(diag)
    display_data(w_show)

    idx = np.random.choice(4000, 100)
    display_data(X[idx])
    display_data(x_hat[idx])
    display_data(h[idx])

