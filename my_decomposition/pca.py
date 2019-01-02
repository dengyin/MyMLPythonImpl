import numpy as np
from scipy.linalg import svd, eig


class MyPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.n_features = X.shape[1]
        _, sigmas, VT = svd(X - self.mean_, full_matrices=False)
        V = VT.T
        V = V[:, np.argsort(sigmas)[::-1]]
        V /= np.linalg.norm(V, axis=0)
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


class MyKernelPCA:
    def __init__(self, n_components=None, kernel='linear', kernel_para=0.1):
        self.__kernels = {'linear': self.__kernel_linear,
                          'rbf': self.__kernel_rbf}
        assert kernel in self.__kernels, 'arg kernel =\'' + kernel + '\' is not available'
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_para = kernel_para

    def __kernel_linear(self, x, y):
        return x.T @ y

    def __kernel_rbf(self, x, y):
        result = np.zeros((x.shape[1], y.shape[1]))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.exp(-self.kernel_para * (x[:, i] - y[:, j]).T @ (x[:, i] - y[:, j]))
        return result

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        K = self.__kernels[self.kernel](X.T, X.T)
        one_M = np.ones((self.n_samples, self.n_samples)) / self.n_samples
        K = K - one_M @ K - K @ one_M + one_M @ K @ one_M
        e_vals, e_vecs = eig(K)
        e_vals, e_vecs = np.real(e_vals), np.real(e_vecs)
        e_vecs /= np.linalg.norm(e_vecs, axis=0)
        e_vecs = e_vecs[:, np.argsort(e_vals)[::-1]] / np.sqrt(np.sort(e_vals)[::-1])
        self.scalings_ = e_vecs
        self.e_vals_ = e_vals
        self.__X = X

    def transform(self, X):
        if not hasattr(self, 'scalings_'):
            raise Exception('Please run `fit` before transform')
        assert X.shape[1] == self.n_features, 'X.shape[1] != self.n_features'
        if self.n_components is None:
            self.n_components = X.shape[1]
        K = self.__kernels[self.kernel](X.T, self.__X.T)
        one_M = np.ones((self.n_samples, self.n_samples)) / self.n_samples
        K = K - one_M @ K - K @ one_M + one_M @ K @ one_M
        return (K @ self.scalings_)[:, :self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
