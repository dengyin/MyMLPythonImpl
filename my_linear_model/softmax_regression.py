import numpy as np

from my_linear_model.tools import cross_entropy_loss
from .base import MyLinearModel


class MySoftmaxRegression(MyLinearModel):
    def __init__(self, stop_threshold=1e-04, steps=0.1, alpha=0.1, normalize=False):

        self.__normalizes = [False, 'l1', 'l2']
        assert normalize in self.__normalizes, 'arg normalize=\'' + normalize + '\' is not available'
        self.normalize = normalize
        if self.normalize == 'l1': assert self.solver == 'coordinate_descent', 'normalize l1 must use coordinate_descent solver'
        self.stop_threshold = stop_threshold
        self.steps = steps
        self.alpha = alpha

        super().__init__()

    def __pre_predict_proba(self, X, w):
        return np.exp(X @ w) / (np.exp(X @ w) @ np.ones((self.n_classes, self.n_classes)))

    def fit(self, X, y):
        self._check_X_y(X, y)
        X = self._trans_X(X)
        self.n_classes = len(np.unique(y))
        y_one_hot = np.eye(self.n_classes)[y]
        if self.normalize is False:
            step_func = lambda X, y_one_hot, w: X.T @ (self.__pre_predict_proba(X, w) - y_one_hot)
        elif self.normalize == 'l2':
            step_func = lambda X, y_one_hot, w: X.T @ (self.__pre_predict_proba(X, w) - y_one_hot) + self.alpha * w
        w = np.random.random((X.shape[1], self.n_classes))
        l_of_w_first = cross_entropy_loss(y_one_hot, self.__pre_predict_proba(X, w))
        while True:
            step = step_func(X, y_one_hot, w)
            w = w - self.steps * step
            l_of_w_last = cross_entropy_loss(y_one_hot, self.__pre_predict_proba(X, w))
            if l_of_w_first - l_of_w_last < self.stop_threshold:
                break
            l_of_w_first = l_of_w_last
        self._w = w

    def predict_proba(self, X):
        return np.exp(super().predict(X)) / (np.exp(super().predict(X)) @ np.ones((self.n_classes, self.n_classes)))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
