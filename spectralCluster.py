import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

n_sample = 200
theta = np.linspace(0, 2 * np.pi, n_sample)
x1, y1 = np.cos(theta), np.sin(theta)+np.random.randn(n_sample)/10
x2, y2 = 2 * np.cos(theta), 2 * np.sin(theta)+np.random.randn(n_sample)/10


x, y = np.hstack((x1, x2)), np.hstack((y1, y2))
X = np.hstack((x1, x2, y1, y2)).reshape(2, -1).T

dist = cdist(X, X)
sigma = .01
A = np.exp(-dist**2 / sigma)
np.fill_diagonal(A, 0)
D = np.diag(np.sum(A, 1))
L = D - A
u, s, vh = np.linalg.svd(L)
idx = u[:, -2] > np.median(u[:, -2])
print(u[:, -2])

fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(x1, y1, c='r')
ax.scatter(x2, y2, c='g')

ax = fig.add_subplot(122)
ax.scatter(X[idx, 0], X[idx, 1], c='r')
ax.scatter(X[~idx, 0], X[~idx, 1], c='g')
plt.show()

