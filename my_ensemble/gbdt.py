from abc import abstractmethod

import numpy as np

from my_tree import MyDecisionTreeRegressor


def convert_to_one_hot(y, C):
    return np.eye(C)[y]


class BaseBoosting:
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    def _predict(self, X):
        if not hasattr(self, '_estimators'):
            raise Exception('Please run `fit` before predict')
        result = 0
        for base_estimator in self._estimators:
            result += base_estimator.predict(X) * self.learning_rate
        return result


class MyGradientBoostingClassifier(BaseBoosting):
    def __init__(
            self,
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        super().__init__()

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_class = len(self.classes)
        _y = convert_to_one_hot(y, self.n_class)
        self._estimators = {c: [] for c in self.classes}
        for c in self.classes:
            base_estimator = MyDecisionTreeRegressor(
                self.max_depth, self.min_samples_split, self.min_samples_leaf)
            base_estimator.fit(X, np.log(np.where(
                _y[:, c] == 1, 1 - 1e-9, 1e-9) / (1 - np.where(_y[:, c] == 1, 1 - 1e-9, 1e-9))))
            self._estimators[c].append(base_estimator)
        while len(self._estimators[0]) < self.n_estimators:
            for c in self.classes:
                base_estimator = MyDecisionTreeRegressor(
                    self.max_depth, self.min_samples_split, self.min_samples_leaf)
                grad = self._calc_grad(
                    _y[:, c].ravel(), self.predict_proba(X)[:, c].ravel())
                base_estimator.fit(X, -grad)
                self._estimators[c].append(base_estimator)
            print(
                "进度:{0}%".format(
                    round(
                        (len(
                            self._estimators[0]) +
                         1) *
                        100 /
                        self.n_estimators)),
                end='\r')

    def _sigmod(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, X):
        denominator = np.sum(np.exp(X), axis=1)
        denominator = np.column_stack([denominator for _ in range(X.shape[1])])
        return np.exp(X) / denominator

    def _calc_grad(self, y_true, y_pred):
        return y_pred - y_true

    def predict_proba(self, X):
        result = []
        for c, estimators in self._estimators.items():
            result.append(0)
            for base_estimator in estimators:
                result[c] += base_estimator.predict(X) * self.learning_rate
        result = self._softmax(np.column_stack(result))
        return result

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).ravel()


class MyGradientBoostingRegressor(BaseBoosting):
    def __init__(
            self,
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        super().__init__()

    def fit(self, X, y):
        self._estimators = []
        base_estimator = MyDecisionTreeRegressor(
            self.max_depth, self.min_samples_split, self.min_samples_leaf)
        base_estimator.fit(X, y)
        self._estimators.append(base_estimator)
        while len(self._estimators) < self.n_estimators:
            base_estimator = MyDecisionTreeRegressor(
                self.max_depth, self.min_samples_split, self.min_samples_leaf)
            grad = self._calc_grad(y.ravel(), self.predict(X).ravel())
            base_estimator.fit(X, -grad)
            self._estimators.append(base_estimator)
            print(
                "进度:{0}%".format(
                    round(
                        (len(
                            self._estimators) +
                         1) *
                        100 /
                        self.n_estimators)),
                end='\r')

    def _calc_grad(self, y_true, y_pred):
        return y_pred - y_true

    def predict(self, X):
        return super()._predict(X)
