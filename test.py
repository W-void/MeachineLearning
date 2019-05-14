import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('creditcard.csv', low_memory=False)

x = df.iloc[:, :-1].values
y = df['Class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

space_num = 5


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

    def cal_ent(self, y=None):
        ent = 0
        num = y.shape[0]
        if num != 0:
            p0, p1 = np.sum(y == 0) / num, np.sum(y == 1) / num
            if p0 * p1 != 0:
                ent = -(p0 * np.log(p0) + p1 * np.log(p1))
        return ent

    def calGain(self, col, x=None, y=None, space_num=5):
        Gain = 0
        feature = x[:, col]
        min, max = np.min(feature), np.max(feature)
        v = np.linspace(min, max, space_num)
        for i in range(1, space_num):
            idx = np.where((feature < v[i]) & (feature > v[i - 1]))
            #print(y.shape, x.shape)
            Gain = Gain + len(idx) / x.shape[0] * self.cal_ent(y[idx])
        return Gain

    def train(self, x=None, y=None, node=None, depth=None, space_num=5):
        if depth == 0:
            return
        if y.shape[0] < 5:
            return
        counts = np.bincount(y)
        if np.max(counts) == y.shape[0]:
            #node.label = y[0]
            #node.x, node.y = x, y
            return
        Gain = []
        for col in range(x.shape[1]):
            Gain.append(self.calGain(col, x, y, space_num))
        featurn_idx = Gain.index(max(Gain))
        feature = x[:, featurn_idx]
        value = np.linspace(np.min(feature), np.max(feature), space_num)
        for i in range(1, space_num):
            idx = np.where((feature < value[i]) & (feature > value[i - 1]))
            new_x, new_y = x[idx, :].reshape((-1, 30)), y[idx]
            #print(new_y.shape, new_x.shape)
            if new_y.shape[0] == 0:
                return
            counts = np.bincount(new_y)
            y_label = np.argmax(counts)
            child = Node(new_x, new_y, y_label, featurn_idx, [value[i-1], value[i]])
            node.append(child)
            self.train(new_x, new_y, child, depth-1)

    def fit(self, x=None, y=None, depth=3):
        self.train(x, y, self.tree, depth)


dt = DTree()
ros = RandomOverSampler(random_state=42)
x_train, y_train = ros.fit_sample(x_train, y_train)
dt.fit(x_train, y_train)
y_pred = dt.tree.batch_predict(x_test)
print(np.sum(y_pred))

np.savetxt('y_pred', y_pred)
acc = np.sum(y_pred == y_test) / len(y_test)
print(acc)

idx = np.where(y_pred == 1)
idx2 = np.where(y_test == 1)
print(idx, idx2, len(idx[0]), len(idx2[0]))