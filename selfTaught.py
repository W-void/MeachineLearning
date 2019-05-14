import SoftmaxRe
import Autoencoder
import scipy.io as scio
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    # 读入数据
    path = 'F:\\UFLDL\\data_digits.mat'
    data = scio.loadmat(path)
    X = data['X']
    Y = data['y']

    # 用5-9训练自编码网络，0-4验证
    idx_unlabel = Y.flatten() > 4  # Y是(5000, 1)，所以需要flatten
    idx_label = Y.flatten() < 5
    x_l, y_l, x_u = X[idx_label], Y[idx_label], X[idx_unlabel]
    enc = OneHotEncoder()
    y_l = enc.fit_transform(y_l).toarray()

    # 生成0-4的新特征，只用新特征分类以检验新特征的有效性
    w, _, _ = Autoencoder.auto_encoder(x_u, layer_size=100, iter=200, p=0.2, alpha=1, lambd=0, beta=1)
    x_l = np.hstack((np.ones((x_l.shape[0], 1)), x_l))  # 加偏置
    a = sigmoid(x_l @ w)
    # x_l = np.hstack((x_l, a))

    x_train, x_test, y_train, y_test = train_test_split(a, y_l, test_size=0.2, random_state=1)
    w = SoftmaxRe.softmax_regression(x_train, y_train, n_iters=50)
    y_pred = SoftmaxRe.predict(x_test, w)
    print('ACC_test', accuracy_score(y_test, y_pred))
