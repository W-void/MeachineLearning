import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def de_sigmoid(z):
    return z * (1 - z)


def nn(x, y, alpha=1, n_iters=10, batch=128):
    (m, n) = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    np.random.seed(10)
    w_1 = np.random.rand(n + 1, n)
    w_2 = np.random.rand(n + 1, 1)
    hidden_layer = np.random.rand(m, n + 1)
    hidden_layer[:, 0] = 1
    y_pred = np.ones((m, 1))

    for _ in range(n_iters):
        i = 0
        while i < m:
            end = min(i + batch, m)
            x_batch, y_batch = x[i:end], y[i:end]

            # 前向传播
            hidden_layer[i:end, 1:] = sigmoid(np.dot(x_batch, w_1))
            y_pred[i:end] = sigmoid(np.dot(hidden_layer[i:end], w_2))

            # 反向传播
            y_ = y_pred[i:end]
            hidden_layer_ = hidden_layer[i:end]
            delta_2 = (y_batch - y_) * de_sigmoid(y_)
            grad_2 = np.dot(hidden_layer_.T, delta_2)

            delta_1 = np.dot(delta_2, w_2[1:].T) * de_sigmoid(hidden_layer_[:, 1:])
            grad_1 = np.dot(x_batch.T, delta_1)

            w_2 += alpha * grad_2
            w_1 += alpha * grad_1

            i = end

        y_pred = sigmoid(np.dot(sigmoid(np.hstack((np.ones((m, 1)), np.dot(x, w_1)))), w_2))
        err = np.array(y_pred - y)
        cost = err.T @ err
        print(cost)

    return [w_1, w_2]


def predict(x, w):
    (m, n) = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    hidden_layer = sigmoid(np.dot(x, w[0]))
    hidden_layer = np.hstack((np.ones((m, 1)), hidden_layer))
    y = sigmoid(np.dot(hidden_layer, w[1]))
    return y > 0.5


bone_data = pd.read_csv('all_bone_info_df.csv')
features_list = list(bone_data.columns)[1:]
features_list.remove('target')
features_list.remove('class_id')

x = bone_data[features_list]
y = bone_data[['target']]

x = scale(x)  # 不标准化很难训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

w = nn(x_train, y_train)
y_pred = predict(x_test, w)
print(accuracy_score(y_test, y_pred))
