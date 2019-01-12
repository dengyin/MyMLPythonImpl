import numpy as np
from sklearn.metrics import mean_squared_error

from my_linear_model.base import MyLinearModel
from my_linear_model.tools import iter_optimization


class MyLinearRegression(MyLinearModel):
    def __init__(self, solver='normal_equation', stop_threshold=0.000001, steps=0.1):
        self.__solvers = {'normal_equation': self.__normal_equation,
                          'newton': self.__newton,
                          'gradient_descent': self.__gradient_descent}
        assert solver in self.__solvers.keys(), 'arg solver=\'' + solver + '\' is not available'
        self.solver = solver
        self.stop_threshold = stop_threshold
        self.steps = steps
        super().__init__()

    def fit(self, X, y):
        self._check_X_y(X, y)
        X = self._trans_X(X)
        self._w = self.__solvers[self.solver](X, y.reshape(-1, 1))
        self.coef_ = self._w.ravel()[:-1]
        self.intercept_ = self._w.ravel()[-1]

    def __normal_equation(self, X, y):
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def __newton(self, X, y):
        return iter_optimization(X, y, lambda X, y, w: np.linalg.inv(2 * X.T @ X) @ (2 * X.T @ X @ w - 2 * X.T @ y),
                                 mean_squared_error, lambda X, w: X @ w, self.steps,
                                 self.stop_threshold)

    def __gradient_descent(self, X, y):
        return iter_optimization(X, y, lambda X, y, w: 2 * X.T @ X @ w - 2 * X.T @ y,
                                 mean_squared_error, lambda X, w: X @ w, self.steps,
                                 self.stop_threshold)


class MyRidge(MyLinearModel):
    def __init__(self, solver='normal_equation', stop_threshold=1e-03, steps=0.1, alpha=0.1):
        self.__solvers = {'normal_equation': self.__normal_equation,
                          'newton': self.__newton,
                          'gradient_descent': self.__gradient_descent}
        assert solver in self.__solvers.keys(), 'arg solver=\'' + solver + '\' is not available'
        self.solver = solver
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

    def __normal_equation(self, X, y):
        i = np.eye(X.shape[1], X.shape[1])
        return np.linalg.inv(X.T @ X + self.alpha * i) @ X.T @ y

    def __newton(self, X, y):
        i = np.eye(X.shape[1], X.shape[1])
        return iter_optimization(X, y, lambda X, y, w: np.linalg.inv(2 * X.T @ X + 2 * self.alpha * i) @ \
                                                       (2 * X.T @ X @ w - 2 * X.T @ y + 2 * self.alpha * w),
                                 mean_squared_error, lambda X, w: X @ w, self.steps,
                                 self.stop_threshold)

    def __gradient_descent(self, X, y):
        return iter_optimization(X, y, lambda X, y, w: (2 * X.T @ X @ w - 2 * X.T @ y + 2 * self.alpha * w),
                                 mean_squared_error, lambda X, w: X @ w, self.steps,
                                 self.stop_threshold)


class MyLasso(MyLinearModel):
    def __init__(self, solver='coordinate_descent', stop_threshold=1e-03, steps=0.1, alpha=0.1):
        self.__solvers = {'coordinate_descent': self.__coordinate_descent}
        assert solver in self.__solvers.keys(), 'arg solver=\'' + solver + '\' is not available'
        self.solver = solver
        self.stop_threshold = stop_threshold
        self.steps = steps
        self.alpha = alpha
        super().__init__()

    def fit(self, X, y):
        self._check_X_y(X, y)
        X = self._trans_X(X)
        self._w = self.__solvers[self.solver](X, y.reshape((-1, 1)))
        self.coef_ = self._w.ravel()[:-1]
        self.intercept_ = self._w.ravel()[-1]

    def __pre_predict(self, X):
        return X @ self._w

    def __coordinate_descent(self, X, y):
        self._w = np.random.random((X.shape[1], 1))
        l_of_w_first = mean_squared_error(y, self.__pre_predict(X))
        n = X.shape[1]
        while True:
            for i in range(n):
                step = X[:, i].T @ (X @ self._w - y) + self.alpha * np.sign(self._w[i, :])
                self._w[i, :] = self._w[i, :] - self.steps * step

            l_of_w_last = mean_squared_error(y, self.__pre_predict(X))
            if l_of_w_first - l_of_w_last < self.stop_threshold:
                return self._w
            l_of_w_first = l_of_w_last


class MyBayesianLinearRegression(MyLinearModel):
    def __init__(self, alpha=0, beta=1):
        self.alpha = alpha
        self.beta = beta
        super().__init__()

    def fit(self, X, y):
        self._check_X_y(X, y)
        X = self._trans_X(X)
        n_samples, n_features = X.shape
        self._sigma_w = np.linalg.inv(self.alpha * np.eye(n_features) + self.beta * X.T @ X)
        self._nue_w = self.beta * self._sigma_w @ X.T @ y.reshape((-1, 1))

    def predict(self, X):
        # before predict, you must run fit func.
        if not hasattr(self, '_nue_w'):
            raise Exception('Please run `fit` before predict')

        X = np.column_stack((X, np.ones((X.shape[0], 1))))

        if X.shape[1] != self._nue_w.shape[0]:
            shape_err = 'X.shape[1]:' + str(X.shape[1] - 1) + " neq n_feature:" + str(self._nue_w.shape[0] - 1)
            raise AssertionError(shape_err)
            del shape_err

        return np.column_stack((X @ self._nue_w, np.diag(1 / self.beta + X @ self._sigma_w @ X.T).reshape(-1, 1)))


class MyGaussianProcessRegression(MyLinearModel):
    def __init__(self, sigma_w=1, sigma_e=0, kernel='linear', kernel_para=0.1):
        self.__kernels = {'linear': self.__kernel_linear,
                          'rbf': self.__kernel_rbf}
        assert kernel in self.__kernels, 'arg kernel =\'' + kernel + '\' is not available'
        self.kernel = kernel
        self.kernel_para = kernel_para
        self.sigma_w = sigma_w
        self.sigma_e = sigma_e
        super().__init__()

    def __kernel_rbf(self, x, y):
        result = np.zeros((x.shape[1], y.shape[1]))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.exp(-self.kernel_para * (x[:, i] - y[:, j]).T @ (x[:, i] - y[:, j]))
        return result

    def __kernel_linear(self, x, y):
        return x.T @ y

    def fit(self, X, y):
        self._check_X_y(X, y)
        n_samples, n_features = X.shape
        X = self._trans_X(X)
        self.__X_train = X
        self.__y_train = y.reshape((-1, 1))
        self.__C_train = self.sigma_e * np.eye(n_samples) + \
                         self.sigma_w * self.__kernels[self.kernel](X.T, X.T)
        self.a_ = np.linalg.inv(self.__C_train) @ self.__y_train

    def predict(self, X):
        if not hasattr(self, 'a_'):
            raise Exception('Please run `fit` before predict')
        X = self._trans_X(X)
        kT = self.__kernels[self.kernel](X.T, self.__X_train.T)
        return (self.a_.T @ kT.T).ravel()
