import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def de_sigmoid(z):
    return z * (1 - z)


def ZCAwhiten(x):
    m, n = x.shape
    x_mean = np.mean(x, 0).reshape((1, -1))
    cov = x.T @ x / m - x_mean.T @ x_mean
    w, v = np.linalg.eig(cov)
    # x_whiten = v / w @ v.T @ x.T
    x_whiten = x @ v / np.sqrt(w) @ v.T
    return x_whiten


def linearDe(x, n_iter=400, layer_size=400, p=0.035, alpha=0.001, beta=1):
    m, n = x.shape
    x_ = np.hstack((np.ones((m, 1)), x))
    np.random.seed(5)
    low = np.sqrt(6 / (n + layer_size))
    w1 = np.random.uniform(-low, low, (n + 1, layer_size))
    w2 = np.random.uniform(-low, low, (layer_size + 1, n))

    for i in range(n_iter):
        # 前向传播
        h = sigmoid(x_ @ w1)
        h_ = np.hstack((np.ones((h.shape[0], 1)), h))
        x_hat = h_ @ w2

        # 梯度检查
        if i == -1:
            epsilon = np.zeros_like(w2)
            epsilon[1, 1] = 1e-4
            x_hat = h_ @ w2
            x_hat1 = h_ @ (w2 + epsilon)
            x_hat2 = h_ @ (w2 - epsilon)
            p_hat = np.mean(h, 0)
            kl = -np.sum(p * np.log(p_hat) + (1 - p) * np.log(1 - p_hat))
            err1 = x_hat1 - x
            cost1 = (np.trace(err1 @ err1.T) / 2 + beta * kl) / m
            err2 = x_hat2 - x
            cost2 = (np.trace(err2 @ err2.T) / 2 + beta * kl) / m

            err = x_hat - x
            delta2 = err
            grad2 = h_.T @ delta2 / m

            print(grad2[1, 1] - ((cost1 - cost2)/(2 * epsilon[1, 1])))


        # 计算损失
        p_hat = np.mean(h, 0)
        kl = -np.sum(p * np.log(p_hat) + (1-p) * np.log(1-p_hat))
        err = x_hat - x
        cost = (np.trace(err.T @ err) / 2 + beta * kl) / m
        if i % 1 == 0:
            print(i, ' : ', cost)


        # 反向传播
        delta2 = err
        grad2 = alpha * h_.T @ delta2 / m

        delta1 = delta2 @ w2[1:].T + beta * (-p/p_hat + (1-p)/(1-p_hat)).reshape(1, -1) * de_sigmoid(h)
        grad1 = 4 * alpha * x_.T @ delta1 / m

        w2 -= grad2
        w1 -= grad1

    return w1, h, x_hat


def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):
    """ Add the weights as a matrix of images """

    figure, axes = plt.subplots(nrows=hid_patch_side,
                                              ncols=hid_patch_side)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_W1[index].reshape(3, vis_patch_side, vis_patch_side).T,
                             interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    plt.show()


if __name__ == '__main__':
    path = 'F:/实习/UFLDL/stlSampledPatches.mat'
    data = scio.loadmat(path)

    x = data['patches'].T
    x = ZCAwhiten(x)

    layer_size = 8
    w, h, x_hat = linearDe(x, n_iter=50, layer_size=layer_size*layer_size, p=0.1, alpha=0.1, beta=1)

    visualizeW1(w[1:, :25].T, 8, 5)
    np.save('w_linearDecode.npy', arr=w)
