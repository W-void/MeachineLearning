import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as scio
import linearDecode
import SoftmaxRe
from sklearn.preprocessing import OneHotEncoder, scale
from scipy import signal


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def avePool(x, size):
    fil = np.ones((size, size)) / size**2
    img_width, _, channelNum, imgNum = x.shape
    y_width = img_width // size
    y = np.zeros((y_width, y_width, channelNum, imgNum))
    for i in range(y_width):
        for j in range(y_width):
            tmp = x[i*size : (i+1)*size, j*size : (j+1)*size, :, :]
            tmp = np.mean(tmp, 0)
            tmp = np.mean(tmp, 0)
            y[i, j] = tmp

    return y


def ZCAwhiten(x):
    a, _, c, m = x.shape
    x = x.reshape(-1, m)
    x_mean = np.mean(x, 1).reshape(-1, 1)
    x -= x_mean
    # inn = x.T @ x
    # w, v = np.linalg.eig(inn)
    u, w, vh = np.linalg.svd(x)
    epsilon = np.ones((u.shape[0])) * 1e-6
    epsilon[:w.shape[0]] = w
    x_whiten = u / epsilon @ u.T @ x
    return x_whiten.reshape(a, a, c, m)


if __name__ == '__main__':
    path = 'F:\\UFLDL\\stlSubset'
    data_test = scio.loadmat(path + '\\stlTestSubset.mat')
    data_train = scio.loadmat(path + '\\stlTrainSubset.mat')

    x_train, x_test, y_train, y_test = data_train['trainImages'], data_test['testImages'], \
                                       data_train['trainLabels'], data_test['testLabels']

    x_train, x_test = ZCAwhiten(x_train), ZCAwhiten(x_test)
    # x_train = scale(x_train.reshape(-1, x_train.shape[-1]).T).reshape(64, 64, 3, -1)
    # x_test = scale(x_test.reshape(-1, x_test.shape[-1]).T).reshape(64, 64, 3, -1)

    w_ld = np.load('w_linearDecode.npy')
    # w_ld = np.load('opt_param.npy')
    b_ld, w_ld = w_ld[0], w_ld[1:]
    kernelSize2, featureNum = w_ld.shape
    kernelSize = np.sqrt(kernelSize2 / 3)
    w_ld = w_ld.T.reshape(25, 3, 8, 8, order='C')
    # w = np.zeros((25, 3, 8, 8))
    # for i in range(featureNum):
    #     w[i] = w_ld[:, i].reshape(3, 8, 8)

    channelNum = 3
    img_width, _, _, train_num = x_train.shape

    h = np.zeros((img_width, img_width, featureNum, train_num))
    for t in range(train_num):
        if t % 100 == 0:
            print(t)
        img = x_train[:, :, :, t]
        for f in range(featureNum):
            w = np.rot90(w_ld[f], 2)
            res = cv2.filter2D(img, ddepth=-1, kernel=w.T)
            # res = signal.convolve2d(img, w.T)
            res = np.sum(res, -1) + b_ld[f]
            h[:, :, f, t] = sigmoid(res)

    left, right = int(kernelSize // 2), int(img_width - kernelSize // 2)
    h = h[left:right+1, left:right+1, :, :]
    h = avePool(h, size=19)

    enc = OneHotEncoder()
    y_train = enc.fit_transform(y_train).toarray()

    x_smx = h.T.reshape(train_num, -1)
    # x_smx = scale(x_smx)
    w_smx = SoftmaxRe.softmax_regression(x_smx, y_train, alpha=1e-5, n_iters=100, lambd=1e-3)

