import numpy as np


class MyFactorAnalysis:
    def __init__(self, n_components=None, tol=1e-7, max_iter=1000):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape
        self.n_features = n_features
        if self.n_components is None:
            self.n_components = n_features
        # 初始化参数
        self.mean_ = X.mean(axis=0).reshape((-1, 1))
        w = np.random.random((n_features, self.n_components))
        mat_sigma = np.eye((n_features))
        log_likelihood = None
        # EM算法
        for _ in range(self.max_iter):
            if log_likelihood is None:
                log_likelihood = float("-inf")
            else:
                new_log_likelihood = - 0.5 * (np.trace(EzzT) + np.trace(
                    (X - Ez @ w.T - self.mean_.reshape(1, -1)) @ np.linalg.pinv(mat_sigma) @ (
                            X - Ez @ w.T - self.mean_.reshape(1, -1)).T
                )) - 0.5 * (n_samples * (self.n_components + n_features)) * np.log(
                    2 * np.pi) - 0.5 * n_samples - 0.5 * n_samples * np.log(np.linalg.det(mat_sigma))
                if new_log_likelihood - log_likelihood < self.tol:
                    break
                log_likelihood = new_log_likelihood
            # E步
            M = np.eye(self.n_components) + w.T @ np.linalg.pinv(mat_sigma) @ w
            Ez = (X - self.mean_.reshape(1, -1)) @ (np.linalg.pinv(M) @ w.T @ np.linalg.pinv(mat_sigma)).T
            EzzT = n_samples * np.linalg.pinv(M) + (Ez.T @ Ez)
            # M步
            w = (X.T - self.mean_) @ Ez @ np.linalg.pinv(EzzT)
            S = (X - self.mean_.reshape(1, -1)).T @ (X - self.mean_.reshape(1, -1)) / n_samples
            mat_sigma = S + (w @ EzzT @ w.T - (X.T - self.mean_) @ Ez @ w.T - w @ Ez.T @ (X - self.mean_.T)) / n_samples
            mat_sigma = np.diag(np.diag(mat_sigma))

        self.scalings_ = (np.linalg.pinv(M) @ w.T @ np.linalg.pinv(mat_sigma)).T

    def transform(self, X):
        if not hasattr(self, 'scalings_'):
            raise Exception('Please run `fit` before transform')
        if not hasattr(self, 'mean_'):
            raise Exception('Please run `fit` before transform')
        assert X.shape[1] == self.n_features, 'X.shape[1] != self.n_features'
        return (X - self.mean_.reshape((1, -1))) @ self.scalings_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
