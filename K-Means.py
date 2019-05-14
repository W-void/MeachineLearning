import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def k_means(x, k=3):
    (m, n) = x.shape
    mean_vec = x[np.random.choice(m, k)]
    new_mean_vec = np.zeros_like(mean_vec)
    dist = np.zeros((k, m))

    # while abs(np.sum(new_mean_vec - mean_vec)) > 5:
    for _ in range(300):
        #print(_)
        for i in range(k):
            d = x - mean_vec[i]
            dist[i] = np.diag(np.dot(d, d.T))
        new_idx = np.argmin(dist, 0)

        for i in range(k):
            new_mean_vec[i] = np.mean(x[new_idx == i], 0)
    return new_mean_vec


# bone_data = pd.read_csv('all_bone_info_df.csv')
# features_list = list(bone_data.columns)[1:]
# features_list.remove('target')
# features_list.remove('class_id')
#
# x = bone_data[features_list]
# y = bone_data['target']

np.random.seed(11)
x = np.zeros((90, 2))
mean = np.array([1, 1])
cov = np.eye(2) * 5
x[:30] = np.random.multivariate_normal(mean, cov, 30)
x[30:60] = np.random.multivariate_normal(mean*10, cov, 30)
x[60:] = np.random.multivariate_normal(mean*20, cov, 30)

kmeans = KMeans(n_clusters=3).fit(x)
print(kmeans.cluster_centers_)
my_kmeans = k_means(x)
print(my_kmeans)
plt.scatter(x[:, 0], x[:, 1])
plt.scatter(my_kmeans[:, 0], my_kmeans[:, 1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
plt.show()
