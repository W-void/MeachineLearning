#%%
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time

#%%
num = 1000
nb = 10

X, color = datasets.samples_generator.make_s_curve(num, random_state=0)

tic = time.process_time()
D = cdist(X, X)

#%%

idx = np.argsort(D, axis=1)

M = np.eye(num)  # M = (I-W)(I-W).T = I - W - W.T - WW.T
tol = 1e-4

for i in range(num):
    idx_ = idx[i, 1:nb+1]
    Z = X[i, :] - X[idx_, :]
    w = np.linalg.pinv(Z @ Z.T + np.eye(nb) * tol * np.trace(Z @ Z.T)) @ np.ones((nb, 1))
    w = w / np.sum(w)

    M[i, idx_] = M[i, idx_] - np.squeeze(w)
    M[idx_, i] = M[idx_, i] - np.squeeze(w)
    x, y = np.meshgrid(idx_, idx_)
    M[x, y] = M[x, y] + w @ w.T

U, _, _ = np.linalg.svd(M)
toc = time.process_time()
#%%

plt.scatter(U[:, -3], U[:, -2], c=color)
plt.show()

print(toc - tic)
