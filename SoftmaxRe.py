import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import scipy.io as scio
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def de_sigmoid(z):
    return z * (1 - z)
np.resize

def softmax(z):
    z_max = np.max(z, 1).reshape(-1, 1)
    z -= z_max
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, 1).reshape(-1, 1)


def de_softmax(z):
    return 1 - z**2


def softmax_regression(x, y, alpha=1e-3, n_iters=100, lambd=1e-3, n_hidden=20, epsilon=1e-10):
    # x是二维数组，每一行是一个样本，每一列是一种特征
    # y是one_hot编码后的标签，也是每一行是一个样本
    (m, n) = x.shape
    num_label = y.shape[1]
    x = np.hstack((np.ones((m, 1)), x))
    np.random.seed(10)

    w = np.random.randn(n + 1, num_label) * np.sqrt(2 / n)
    y_pred = np.zeros_like(y)

    for _ in range(n_iters):
        y_pred = softmax(x @ w)

        weight_decay = np.sum(w[:, 1:].T @ w[:, 1:])
        cost = -np.sum(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon)) / m + .5 * lambd * weight_decay
        if _ % 5 == 0:
            print(_, ' : ', cost)

        w_tmp = w.copy()
        w_tmp[0] = 0  # 好像加了这两句没什么区别
        grad = alpha * x.T @ (y_pred - y) + lambd * w_tmp  # 好像加不加权重衰减都一样。。。
        w -= grad

    print('ACC_train:', accuracy_score(y, y_pred > 0.5))
    return w

    # w_1 = np.random.rand(n + 1, n_hidden) * np.sqrt(2 / n)
    # hidden_layer = np.ones((m, n_hidden + 1))
    # w_2 = np.random.rand(n_hidden + 1, num_label) * np.sqrt(2 / n)
    # y_pred = np.zeros_like(y)
    #
    # for _ in range(n_iters):
    #
    #     # 前向传播
    #     hidden_layer[:, 1:] = sigmoid(x @ w_1)
    #     y_pred = softmax(hidden_layer @ w_2)
    #
    #     # 计算误差
    #     weight_decay = np.sum(w_1[:, 1:].T @ w_1[:, 1:]) + np.sum(w_2[:, 1:].T @ w_2[:, 1:])
    #     cost = -np.sum(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon)) / m + .5 * lambd * weight_decay
    #     if _ % 9 == 0:
    #         print(cost)
    #
    #     # 反向传播
    #     delta_2 = y_pred - y
    #     w_tmp = w_2.copy()
    #     w_tmp[0] = 0
    #     grad_2 = alpha * hidden_layer.T @ delta_2 + lambd * w_tmp
    #
    #     delta_1 = delta_2 @ w_2[1:].T * de_sigmoid(hidden_layer[:, 1:])
    #     w_tmp = w_1.copy()
    #     w_tmp[0] = 0
    #     grad_1 = alpha * x.T @ delta_1 + lambd * w_tmp
    #
    #     w_2 -= grad_2
    #     w_1 -= grad_1
    #
    # print('ACC_train:', accuracy_score(y, y_pred > 0.5))
    # return [w_1, w_2]


def predict(x, w):
    (m, n) = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    # hidden_layer = sigmoid(np.dot(x, w[0]))
    # hidden_layer = np.hstack((np.ones((m, 1)), hidden_layer))
    # y = softmax(np.dot(hidden_layer, w[1]))
    y = softmax(x @ w)
    return y > 0.5


if __name__ == '__main__':
    path = 'F:/dataset/data_digits.mat'
    data = scio.loadmat(path)
    X = data['X']
    Y = data['y']
    print(X.shape, Y.shape)
    enc = OneHotEncoder()
    y = enc.fit_transform(Y).toarray()
    X = scale(X)  # 虽然是图像，有没有这句准确率还是差很多的
    # y = np.zeros((len(Y), 10))
    # for i in range(10):
    #     y[:, i] = np.int32(Y == i).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    w = softmax_regression(x_train, y_train)
    y_pred = predict(x_test, w)
    print('ACC_test', accuracy_score(y_test, y_pred))
