import numpy as np

from .base import MyLinearModel
from .tools import sigma


class MyGaussianDiscriminatAnalysis(MyLinearModel):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        self._check_X_y(X, y)
        m, n = X.shape
        fi = y[y == 1].shape[0] / m
        u1 = X[y == 1, :].mean(axis=0).reshape((-1, 1))
        u0 = X[y == 0, :].mean(axis=0).reshape((-1, 1))
        sigma = (X[y == 1, :].T - u1) @ (X[y == 1, :].T - u1).T + \
                (X[y == 0, :].T - u0) @ (X[y == 0, :].T - u0).T
        sigma /= m
        self.coef_ = (np.linalg.pinv(sigma) @ (u1 - u0)).ravel()
        self.intercept_ = -0.5 * u1.T @ np.linalg.pinv(sigma) @ u1 \
                          + 0.5 * u0.T @ np.linalg.pinv(sigma) @ u0 \
                          + np.log(fi / (1 - fi))
        self._w = np.row_stack((self.coef_.reshape((-1, 1)), self.intercept_))

    def predict_proba(self, X):
        return sigma(super().predict(X))

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)
