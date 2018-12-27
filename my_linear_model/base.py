from abc import abstractmethod

import numpy as np


class MyLinearModel():
    """
    Abstract base class of Linear Model.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        """fit func"""

    def _check_X_y(self, X, y):
        assert len(y.shape) == 1, 'len(y.shape) != 1'
        assert X.shape[0] == y.shape[0], 'X.shape[0]:' + str(X.shape[0]) + "neq y.shape:" + str(y.shape[0])

    def _trans_X(self, X):
        return np.column_stack((X, np.ones((X.shape[0], 1))))

    def predict(self, X):
        # before predict, you must run fit func.
        if not hasattr(self, '_w'):
            raise Exception('Please run `fit` before predict')

        X = np.column_stack((X, np.ones((X.shape[0], 1))))

        if X.shape[1] != self._w.shape[0]:
            shape_err = 'X.shape[1]:' + str(X.shape[1] - 1) + " neq n_feature:" + str(self._w.shape[0] - 1)
            raise AssertionError(shape_err)
            del shape_err

        # `x @ y` == `np.dot(x, y)`
        return X @ self._w
