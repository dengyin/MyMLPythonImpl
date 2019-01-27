from math import pi

import numpy as np


def norm_distribute(x, mean, sigma):
    n = mean.shape[0]
    if np.linalg.det(sigma) == 0:
        sigma += np.eye(n) * 0.01
    p = ((2 * pi) ** (-n / 2)) * (np.linalg.det(sigma) ** (-0.5)) \
        * np.exp(
        -0.5 * (x - mean).T @ np.linalg.inv(sigma) @ (x - mean)
    )
    return p


class MyGaussianMixture:
    def __init__(self, n_componts=1, max_iter=100, tol=1e-3):
        self.n_componts = n_componts
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        m, self.n_features = X.shape
        mean = []
        sigma = []
        for _ in range(self.n_componts):
            # mean.append(X.mean(axis=0).reshape((-1, 1)))
            # sigma.append(np.cov(X.T))
            mean.append(np.random.random((self.n_features, 1)))
            sigma.append(np.eye(self.n_features))
        fi = np.ones(self.n_componts) / self.n_componts
        z = np.zeros((m, self.n_componts))
        oldloglikelyhood = 0
        for _ in range(self.max_iter):
            # E step
            for k in range(self.n_componts):
                z[:, k] = fi[k] * np.diag(norm_distribute(X.T, mean[k], sigma[k])).reshape((1, -1))
            z /= z.sum(axis=1).reshape(-1, 1)

            # M step
            fi = z.sum(axis=0) / m
            for k in range(self.n_componts):
                mean[k] = np.sum(z[:, k].reshape(-1, 1) * X, axis=0).reshape((-1, 1)) / np.sum(z[:, k])
                sigma[k] = ((X.T - mean[k]) @ np.diag(z[:, k]) @ (X.T - mean[k]).T) / np.sum(z[:, k])
            loglikelyhood = np.sum(
                [np.log(np.sum(
                    [fi[k] * norm_distribute(X[i, :].reshape((-1, 1)), mean[k], sigma[k])
                     for k in range(self.n_componts)]
                ))
                    for i in range(m)]
            )
            if np.abs(loglikelyhood - oldloglikelyhood) < self.tol:
                break
            else:
                oldloglikelyhood = loglikelyhood

        self.fi = fi
        self.mean = mean
        self.sigma = sigma

    def predict_proba(self, X):
        m, n = X.shape
        z = np.zeros((m, self.n_componts))
        for k in range(self.n_componts):
            z[:, k] = self.fi[k] * np.diag(norm_distribute(X.T, self.mean[k], self.sigma[k])).reshape((1, -1))
        z /= z.sum(axis=1).reshape(-1, 1)
        return z

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
