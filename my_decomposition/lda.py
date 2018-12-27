import numpy as np
from scipy.linalg import eigh


class MyLinearDiscriminantAnalysis:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        self.n_features = X.shape[1]
        u1 = X[y == 1, :].mean(axis=0).reshape((-1, 1))
        u0 = X[y == 0, :].mean(axis=0).reshape((-1, 1))
        Sb = (u1 - u0) @ (u1 - u0).T
        Sw = (X[y == 1, :].T - u1) @ (X[y == 1, :].T - u1).T + (X[y == 0, :].T - u0) @ (X[y == 0, :].T - u0).T
        e_vals, e_vecs = eigh(Sb, Sw)
        e_vecs = e_vecs[:, np.argsort(e_vals)[::-1]]
        e_vecs /= np.apply_along_axis(np.linalg.norm, 0, e_vecs)
        self.scalings_ = e_vecs

    def transform(self, X):
        if not hasattr(self, 'scalings_'):
            raise Exception('Please run `fit` before transform')
        assert X.shape[1] == self.n_features, 'X.shape[1] != self.n_features'
        if self.n_components is None:
            self.n_components = self.n_features
        return (X @ self.scalings_)[:, :self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
