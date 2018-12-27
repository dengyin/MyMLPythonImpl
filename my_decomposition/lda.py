import numpy as np
from scipy.linalg import eigh


class MyLDA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        u1 = X[y == 1, :].mean(axis=0).reshape((-1, 1))
        u0 = X[y == 0, :].mean(axis=0).reshape((-1, 1))
        Sb = (u1 - u0) @ (u1 - u0).T
        Sw = (X[y == 1, :].T - u1) @ (X[y == 1, :].T - u1).T + (X[y == 0, :].T - u0) @ (X[y == 0, :].T - u0).T
        e_vals, e_vecs = eigh(Sb, Sw)
        e_vecs = e_vecs[:, np.argsort(e_vals)[::-1]]
        e_vecs /= np.apply_along_axis(np.linalg.norm, 0, e_vecs)
        self.scalings_ = e_vecs

    def transform(self, X):
        self.n_components = X.shape[1]
        return (X @ self.scalings_)[:, :self.n_components]
