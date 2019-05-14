from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split


def rf(x, y, n_tree=10, depth=5):
    forest = []
    (m, n) = x.shape
    for i in range(n_tree):
        dt = DecisionTreeClassifier(max_depth=depth)
        idx = np.random.choice(m, m)
        col = np.random.choice(n, int(n/2), replace=False)
        x_train, y_train = x[idx], y[idx]
        x_train[:, col] = 0
        dt.fit(x_train, y_train)

        # compute OOB error
        idx_ = list(set(np.arange(m)) - set(idx))
        x_test, y_test = x[idx_], y[idx_]
        y_pred = dt.predict(x_test)
        # print(y_pred[:10], y_test[:10].flatten())
        acc = metrics.accuracy_score(y_pred, y_test.flatten())
        forest.append([dt, acc])
    return forest

def predict(rf, x):
    y = np.zeros((x.shape[0]))
    for [dt, acc] in rf:
        y += dt.predict(x)
    return (y / len(rf) + 0.5).astype(np.int32)

bone_data = pd.read_csv('all_bone_info_df.csv')
features_list = list(bone_data.columns)[1:]
features_list.remove('class_id')
features_list.remove('target')

x = bone_data[features_list]
y = bone_data[['target']]

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=1)
np.random.seed(10)
clf = rf(x_train, y_train)
y_pred = predict(clf, x_test)
print(metrics.accuracy_score(y_test, y_pred))
