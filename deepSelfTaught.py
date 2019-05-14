import SoftmaxRe
import Autoencoder
import scipy.io as scio
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z+1e-6))


def de_sigmoid(z):
    return z * (1 - z)


def softmax(z):
    z_max = np.max(z, 1).reshape(-1, 1)
    z -= z_max
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, 1).reshape(-1, 1)


def deepTaught(x, w, y, n_iter=100, alpha=0.01):
    w1, w2, w3 = w[0], w[1], w[2]
    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))

    for i in range(n_iter):
        # 前向传播
        h1 = sigmoid(x @ w1)
        h1_ = np.hstack((np.ones((h1.shape[0], 1)), h1))
        h2 = sigmoid(h1_ @ w2)
        h2_ = np.hstack((np.ones((h2.shape[0], 1)), h2))
        y_pred = softmax(h2_ @ w3)

        # 计算损失
        epsilon = 1e-6
        cost = -np.sum(y * np.log(y_pred+epsilon) + (1-y) * np.log(1-y_pred+epsilon)) / m
        if i % 10 == 0:
            print(i, ' : ', cost)

        # 反向传播
        delta3 = y_pred - y
        grad3 = alpha * h2_.T @ delta3

        delta2 = delta3 @ w3[1:].T * de_sigmoid(h2)
        grad2 = 4 * alpha * h1_.T @ delta2

        delta1 = delta2 @ w2[1:].T * de_sigmoid(h1)
        grad1 = 4 * alpha * x.T @ delta1

        w3 -= grad3
        w2 -= grad2
        w1 -= grad1

    return [w1, w2, w3]


def predict(x, w):
    h1 = sigmoid(np.hstack((np.ones((x.shape[0], 1)), x)) @ w[0])

    h2 = sigmoid(np.hstack((np.ones((h1.shape[0], 1)), h1)) @ w[1])

    y_pred = SoftmaxRe.predict(h2, w[2])

    return y_pred

if __name__ == '__main__':
    # 读入数据
    path = 'F:\\UFLDL\\data_digits.mat'
    data = scio.loadmat(path)
    X = data['X']
    Y = data['y']

    enc = OneHotEncoder()
    Y = enc.fit_transform(Y).toarray()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # pre_training
    print('autoEncoder1')
    w1, h1, _ = Autoencoder.auto_encoder(x_train, layer_size=100, iter=100, p=0.2, alpha=1, lambd=0, beta=1)
    print('autoEncoder2')
    w2, h2, _ = Autoencoder.auto_encoder(h1, layer_size=100, iter=50, p=0.2, alpha=1, lambd=0, beta=1)
    idx = np.random.choice(4000, 100)
    Autoencoder.display_data(_[idx])
    print('softmax')
    w3 = SoftmaxRe.softmax_regression(h2, y_train, n_iters=50, alpha=1e-3)

    # fine_tune
    w_all = deepTaught(x_train, [w1, w2, w3], y_train, alpha=1e-4)

    y_pred = predict(x_test, w_all)
    print('ACC_test', accuracy_score(y_test, y_pred))
