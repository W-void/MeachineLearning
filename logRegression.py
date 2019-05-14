import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.io as scio
from sklearn.datasets import load_iris


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def de_sigmoid(y):
    return y * (1-y)


def LR(x, y, alpha=0.001, iterate=1000, lamda=0.01, batch=1000):

    np.random.seed(1)
    m, n = x.shape
    x = np.hstack((x, np.ones((m, 1))))
    y = y.reshape((-1, 1))
    w = np.random.rand(n + 1, 1)

    # gradient checking
    f = sigmoid(np.dot(x, w))
    g = np.sum(((f - y) * x), 0).reshape((-1, 1)) / m + lamda * w
    epsilon = np.zeros_like(w)
    epsilon[1] = 1e-5
    w1, w0 = w + epsilon, w - epsilon
    f1 = sigmoid(np.dot(x, w1))
    f0 = sigmoid(np.dot(x, w0))
    J1 = -np.sum(y * np.log2(f1) + (1-y) * np.log2(1-f1)) / m + lamda * np.dot(w1.T, w1) / 2
    J0 = -np.sum(y * np.log2(f0) + (1-y) * np.log2(1-f0)) / m + lamda * np.dot(w0.T, w0) / 2
    print(g[0, 0] - ((J1 - J0)/(2 * 1e-5)))

    for _ in range(iterate):
        i = 0
        while i < m:
            end = min(m, i + batch)
            x_batch, y_batch = x[i:end], y[i:end]
            f = sigmoid(np.dot(x_batch, w))

            w_tmp = w.copy()
            w_tmp[0] = 0
            delta_w = alpha * np.sum(((f - y_batch) * x_batch), 0).reshape((-1, 1)) / (end-i) + lamda * w_tmp

            w = w - delta_w
            i = i + batch

        f = sigmoid(np.dot(x, w)).flatten()
        if _ % 100 == 0:
            cost = -np.sum(y * np.log2(f + 1e-10) + (1-y) * np.log2(1-f + 1e-10)) + lamda* np.dot(w.T, w)
            # print(cost)
    # print(w)
    return w


def predict(x, w):
    m = x.shape[0]
    x = np.hstack((x, np.ones((m, 1))))
    y = np.dot(x, w)
    return y > 0.5


def cal_ACC(y, y_p):
    return sum(y == y_p) / len(y)


if __name__ == '__main__':
    #Data = pd.read_csv('creditcard.csv').values
    iris = load_iris()
    x, y = iris.data, iris.target
    y = (y > 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    w = LR(x_train, y_train)
    y_p = predict(x_test, w).flatten()
    acc = cal_ACC(y_test, y_p)

    #print(y_test, y_p)
    # print(len(y_test), sum(y_test), sum(y_p))
    print('acc:', acc)