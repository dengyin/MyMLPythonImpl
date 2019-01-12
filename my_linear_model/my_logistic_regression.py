import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import log_loss

from my_linear_model.tools import iter_optimization
from .base import MyLinearModel
from .tools import sigma


class MyLogisticRegression(MyLinearModel):
    def __init__(self, solver='gradient_descent', stop_threshold=1e-04, steps=0.1, alpha=0.1, normalize=False):
        self.__solvers = {'newton': self.__newton,
                          'gradient_descent': self.__gradient_descent,
                          'coordinate_descent': self.__coordinate_descent}
        self.__normalizes = [False, 'l1', 'l2']
        assert solver in self.__solvers.keys(), 'arg solver=\'' + solver + '\' is not available'
        assert normalize in self.__normalizes, 'arg normalize=\'' + normalize + '\' is not available'
        self.normalize = normalize
        self.solver = solver
        if self.normalize == 'l1': assert self.solver == 'coordinate_descent', 'normalize l1 must use coordinate_descent solver'
        self.stop_threshold = stop_threshold
        self.steps = steps
        self.alpha = alpha

        super().__init__()

    def fit(self, X, y):
        self._check_X_y(X, y)
        X = self._trans_X(X)
        self._w = self.__solvers[self.solver](X, y.reshape(-1, 1))
        self.coef_ = self._w.ravel()[:-1]
        self.intercept_ = self._w.ravel()[-1]

    def predict_proba(self, X):
        return sigma(super().predict(X))

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)

    def __newton(self, X, y):
        i = np.eye(X.shape[1], X.shape[1])
        if self.normalize is False:
            step_fun = lambda X, y, w: np.linalg.pinv(
                (X.T @ np.diag((sigma(X @ w) * (1 - sigma(X @ w))).ravel()) @ X)
            ) @ X.T @ (sigma(X @ w) - y)
        elif self.normalize == 'l2':
            step_fun = lambda X, y, w: np.linalg.pinv(
                (X.T @ np.diag((sigma(X @ w) * (1 - sigma(X @ w))).ravel()) @ X) + self.alpha * i
            ) @ (X.T @ (sigma(X @ w) - y) + self.alpha * w)
        return iter_optimization(X, y, step_fun,
                                 log_loss, lambda X, w: sigma(X @ w), self.steps,
                                 self.stop_threshold)

    def __gradient_descent(self, X, y):
        if self.normalize is False:
            step_fun = lambda X, y, w: X.T @ (sigma(X @ w) - y)
        elif self.normalize == 'l2':
            step_fun = lambda X, y, w: X.T @ (sigma(X @ w) - y) + self.alpha * w
        return iter_optimization(X, y, step_fun,
                                 log_loss, lambda X, w: sigma(X @ w), self.steps,
                                 self.stop_threshold)

    def __coordinate_descent(self, X, y):
        w = np.random.random((X.shape[1], 1))
        l_of_w_first = log_loss(y, sigma(X @ w))
        n = X.shape[1]
        while True:
            for i in range(n):
                step = X[:, i].T @ (sigma(X @ w) - y) + self.alpha * np.sign(w[i, :])
                w[i, :] = w[i, :] - self.steps * step

            l_of_w_last = log_loss(y, sigma(X @ w))
            if l_of_w_first - l_of_w_last < self.stop_threshold:
                break
            l_of_w_first = l_of_w_last
        return w


class MyBayesianLogisticRegression(MyLinearModel):
    def __init__(self, alpha=1, solver='gradient_descent', stop_threshold=1e-04, steps=0.1, predict_way='expect'):
        self.__solvers = {'newton': None, 'gradient_descent': None, 'coordinate_descent': None}
        assert solver in self.__solvers.keys(), 'arg solver=\'' + solver + '\' is not available'
        self.solver = solver
        self.stop_threshold = stop_threshold
        self.steps = steps
        self.alpha = alpha
        self.__predict_ways = {'expect': self.__expect, 'monte_carlo': self.__monte_carlo}
        assert predict_way in self.__predict_ways.keys(), 'arg solver=\'' + predict_way + '\' is not available'
        self.predict_way = predict_way
        super().__init__()

    def fit(self, X, y):
        n_samples, n_features = X.shape
        t = MyLogisticRegression(solver=self.solver, stop_threshold=self.stop_threshold, steps=self.steps,
                                 alpha=0.5 / self.alpha, normalize='l2')
        t.fit(X, y)
        self._nue_w = t._w
        self.coef_ = t.coef_
        self.intercept_ = t.intercept_
        del t
        X = np.column_stack((X, np.ones((X.shape[0], 1))))
        self._sigma_w = np.eye(n_features + 1) / self.alpha + X.T @ np.diag(y * (1 - y)) @ X

    def __expect(self, X):
        return sigma(X @ self._nue_w)

    def __monte_carlo(self, X):
        n_samples, n_features = X.shape
        result = np.zeros((n_samples, 1))
        for i in range(10000):
            w = np.random.uniform(low=-1e+3, high=1e+3, size=(n_features, 1))
            result += multivariate_normal.pdf(w.ravel(), mean=self._nue_w.ravel(), cov=self._sigma_w) * sigma(X @ w)
        return result / n_samples

    def predict_proba(self, X):
        # before predict, you must run fit func.
        if not hasattr(self, '_nue_w'):
            raise Exception('Please run `fit` before predict')

        X = np.column_stack((X, np.ones((X.shape[0], 1))))

        if X.shape[1] != self._nue_w.shape[0]:
            shape_err = 'X.shape[1]:' + str(X.shape[1] - 1) + " neq n_feature:" + str(self._nue_w.shape[0] - 1)
            raise AssertionError(shape_err)
            del shape_err

        return self.__predict_ways[self.predict_way](X)

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)
