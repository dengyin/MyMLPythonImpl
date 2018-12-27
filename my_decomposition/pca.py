import numpy as np
from scipy.linalg import svd


class MyPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.n_features = X.shape[1]
        _, sigmas, VT = svd(X - self.mean_, full_matrices=False)
        V = VT.T
        V = V[:, np.argsort(sigmas)[::-1]]
        # V /= np.apply_along_axis(np.linalg.norm, 0, V)
        self.scalings_ = V

    def transform(self, X):
        if not hasattr(self, 'scalings_'):
            raise Exception('Please run `fit` before transform')
        if not hasattr(self, 'mean_'):
            raise Exception('Please run `fit` before transform')
        assert X.shape[1] == self.n_features, 'X.shape[1] != self.n_features'
        if self.n_components is None:
            self.n_components = self.n_features
        return ((X - self.mean_) @ self.scalings_)[:, :self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
