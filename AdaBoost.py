from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, x=None, y=None, label=None, feature=None, value=None):
        self.label = label
        self.x = x
        self.y = y
        self.value = value
        self.feature = feature
        self.child = []

    def append(self, node):
        self.child.append(node)

    def predict(self, x):
        if len(self.child) == 0:
            return self.label
        for c in self.child:
            #print(c.value[0], c.value[1], len(self.child), c.feature, x[c.feature])
            if ((x[c.feature]>c.value[0]) and (x[c.feature]<c.value[1])):
                return c.predict(x)
    def batch_predict(self, x_test):
        y_pred = np.zeros((x_test.shape[0]))
        for i in range(x_test.shape[0]):
            x = x_test[i, :]
            y_pred[i] = self.predict(x)
        return y_pred


class DTree:
    def __init__(self):
        self.tree = Node()

    def cal_ent(self, y=None, w=None):
        ent = 0
        num = y.shape[0]
        if num != 0:
            # p0, p1 = np.sum(y == 0) / num, np.sum(y == 1) / num
            p0, p1 = w[y == -1].sum(), w[y == 1].sum()
            if p0 * p1 != 0:
                ent = -(p0 * np.log(p0) + p1 * np.log(p1))
        return ent

    def calGain(self, col, x=None, y=None, space_num=5, w=None):
        Gain = 0
        feature = x[:, col]
        min, max = np.min(feature), np.max(feature)
        v = np.linspace(min, max, space_num)
        for i in range(1, space_num):
            idx = np.where((feature < v[i]) & (feature > v[i - 1]))
            #print(y.shape, x.shape)
            Gain = Gain + len(idx) / x.shape[0] * self.cal_ent(y[idx], w[idx])
        return Gain

    def train(self, x=None, y=None, node=None, depth=None, space_num=5, w=None):
        if depth == 0:
            return
        if y.shape[0] < 5:
            return
        if len(set(y)) == 1:
            #node.label = y[0]
            #node.x, node.y = x, y
            return
        Gain = []
        for col in range(x.shape[1]):
            Gain.append(self.calGain(col, x, y, space_num, w=w))
        featurn_idx = Gain.index(max(Gain))
        feature = x[:, featurn_idx]
        value = np.linspace(np.min(feature), np.max(feature), space_num)
        for i in range(1, space_num):
            idx = np.where((feature < value[i]) & (feature > value[i - 1]))
            new_x, new_y = x[idx, :], y[idx]
            #print(new_y.shape, new_x.shape)
            if new_y.shape[0] == 0:
                return

            # counts = np.bincount(np.int64(new_y))
            # y_label = np.argmax(counts)
            y_label = pd.value_counts(new_y).index[0]
            child = Node(new_x, new_y, y_label, featurn_idx, [value[i-1], value[i]])
            node.append(child)
            self.train(new_x, new_y, child, depth-1, w[idx])

    def fit(self, x=None, y=None, depth=1, w=None):
        self.train(x, y, self.tree, depth, w=w)


def adaboost(x, y, n_estimators=10):
    ada = []
    (m, n) = x.shape
    y = y.flatten()
    y[y == 0] = -1
    f = np.zeros_like(y)
    d = np.ones((m)) / m
    for i in range(n_estimators):
        # dt = DecisionTreeClassifier(max_depth=depth)
        dt = DTree()
        dt.fit(x, y, w=d)
        y_pred = dt.tree.batch_predict(x)
        e = np.sum(d * (y_pred != y))
        alpha = 0.5 * np.log2((1-e) / e)
        d = d * np.exp(-alpha * y * y_pred)
        d = d / np.sum(d)
        ada.append([dt, alpha])
        f += alpha * y_pred
    return f > 0, ada

bone_data = pd.read_csv('all_bone_info_df.csv')
features_list = list(bone_data.columns)[1:]
features_list.remove('class_id')
features_list.remove('target')

x = bone_data[features_list]
y = bone_data[['target']]

y_pred, _ = adaboost(x.values, y.values)
print(accuracy_score(y, y_pred))
