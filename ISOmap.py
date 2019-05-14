from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
Axes3D

n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 10

D = cdist(X, X)
# inner = X @ X.T
# D0 = np.diag(inner) + np.diag(inner)[:, None] - 2*inner

idx = np.argsort(D)[:, n_neighbors+1:]
idx2 = np.where(idx > -1)
D[idx2[0], idx[idx2]] = 1e5

'''
for i in range(n_points):
    D[i, idx[i]] = 1e5
'''

floyd = floyd_warshall(csgraph=D)

# MDS
D2 = floyd**2
B = -.5 * (np.sum(D2)/n_points**2 - np.sum(D2, 0)/n_points - np.sum(D2, 1).reshape(-1, 1)/n_points + D2)

u, s, vh = np.linalg.svd(B)
z = u[:, :2] @ np.diag(np.sqrt(s[:2]))

# plot
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(122)
plt.scatter(z[:, 0], z[:, 1], c=color, cmap=plt.cm.Spectral)
plt.show()

