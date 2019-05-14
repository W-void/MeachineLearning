import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def pca(x, n_feature):
    mean_x = np.mean(x, 0)
    x -= mean_x
    eig, vec = np.linalg.eig(np.dot(x.T, x))
    idx = np.argsort(-eig)
    W = vec[:, idx[:n_feature]]
    new_x = np.dot(x, W)
    return new_x


bone_data = pd.read_csv('all_bone_info_df.csv')
features_list = list(bone_data.columns)[1:]
features_list.remove('class_id')
features_list.remove('target')

x = bone_data[features_list]
y = bone_data[['target']]
PCA_x = pca(x.values, 10)

x_train, x_test, y_train, y_test = train_test_split(PCA_x, y, test_size=0.2, random_state=1)
gbdt = GradientBoostingClassifier(random_state=3)
gbdt.fit(x_train, y_train)
y_pred = gbdt.predict(x_test)
# print(y_pred.dtype, y_test.values.dtype)
print("accuracy: %.4g" % (metrics.accuracy_score(y_test, y_pred)))
print(len(features_list))