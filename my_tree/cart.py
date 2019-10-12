from abc import abstractmethod

import numpy as np


class BaseTree:
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        """fit func"""

    def _predict(self, X):
        # before predict, you must run fit func.
        if not hasattr(self, '_tree'):
            raise Exception('Please run `fit` before predict')
        result = []
        for x in X:
            dic = self._tree
            while 'feature' in dic:
                dic = dic['right'] if x[dic['feature']] >= dic['value'] else dic['left']
            result.append(dic['leaf'])
        return result


class DecisionTreeClassifier(BaseTree):
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        super().__init__()

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_class = len(self.classes)
        self._depth = 0
        self._tree = self._create_tree(X, y)

    def predict_proba(self, X):
        result = self._predict(X)
        return np.column_stack(result) if self.n_class == 2 else np.row_stack(result)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba < 0.5, 0, 1).ravel() if self.n_class == 2 else np.argmax(proba, axis=1).ravel()

    def _create_tree(self, X, y, depth=0):
        best_feature, best_value = self._choose_best_feature_value(X, y)
        if depth > self._depth:
            self._depth = depth

        if best_feature == None or self._depth >= self.max_depth:
            leaf = np.mean(y) if self.n_class == 2 else np.array([len(y[y == c]) / len(y) for c in self.classes])
            tree = {'leaf': leaf}
        else:
            left_X, right_X, left_y, right_y = self._split_data(X, y, best_feature, best_value)
            tree = {'feature': best_feature, 'value': best_value}
            tree['left'] = self._create_tree(left_X, left_y, depth + 1)
            tree['right'] = self._create_tree(right_X, right_y, depth + 1)
        return tree

    def _calc_gini(self, y):
        if len(y) == 0:
            return 0
        gini = 1.0
        for c in self.classes:
            gini -= (len(y[y == c]) / len(y)) ** 2
        return gini

    def _split_data(self, X, y, feature_index, value):
        left_X = X[X[:, feature_index] < value]
        right_X = X[X[:, feature_index] >= value]
        left_y = y[X[:, feature_index] < value]
        right_y = y[X[:, feature_index] >= value]
        return left_X, right_X, left_y, right_y

    def _choose_best_feature_value(self, X, y):
        best_feature, best_value = None, None
        best_gain = 0
        if len(y) < self.min_samples_split:
            return best_feature, best_value
        gini = self._calc_gini(y)
        for f in range(X.shape[1]):
            for v in np.unique(X[:, f]):
                left_X, right_X, left_y, right_y = self._split_data(X, y, f, v)
                gini_gain = gini - self._calc_gini(left_y) * len(left_y) / len(y) - self._calc_gini(right_y) * len(
                    right_y) / len(y)
                if gini_gain > best_gain and np.min([len(left_y), len(right_y)]) >= self.min_samples_leaf:
                    best_gain = gini_gain
                    best_feature, best_value = f, v
        return best_feature, best_value


class DecisionTreeRegressor(BaseTree):
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        super().__init__()

    def fit(self, X, y):
        self._depth = 0
        self._tree = self._create_tree(X, y)

    def predict(self, X):
        result = self._predict(X)
        return np.column_stack(result).ravel()

    def _create_tree(self, X, y, depth=0):
        best_feature, best_value = self._choose_best_feature_value(X, y)
        if depth > self._depth:
            self._depth = depth

        if best_feature == None or self._depth >= self.max_depth:
            leaf = np.mean(y)
            tree = {'leaf': leaf}
        else:
            left_X, right_X, left_y, right_y = self._split_data(X, y, best_feature, best_value)
            tree = {'feature': best_feature, 'value': best_value}
            tree['left'] = self._create_tree(left_X, left_y, depth + 1)
            tree['right'] = self._create_tree(right_X, right_y, depth + 1)
        return tree

    def _calc_mse(self, y):
        return np.sum((y - np.mean(y)) ** 2)

    def _split_data(self, X, y, feature_index, value):
        left_X = X[X[:, feature_index] < value]
        right_X = X[X[:, feature_index] >= value]
        left_y = y[X[:, feature_index] < value]
        right_y = y[X[:, feature_index] >= value]
        return left_X, right_X, left_y, right_y

    def _choose_best_feature_value(self, X, y):
        best_feature, best_value = None, None
        best_mse = float('inf')
        if len(y) < self.min_samples_split:
            return best_feature, best_value
        for f in range(X.shape[1]):
            for v in np.unique(X[:, f]):
                left_X, right_X, left_y, right_y = self._split_data(X, y, f, v)
                cur_mse = self._calc_mse(left_y) + self._calc_mse(right_y)
                if cur_mse < best_mse and np.min([len(left_y), len(right_y)]) >= self.min_samples_leaf:
                    best_mse = cur_mse
                    best_feature, best_value = f, v
        return best_feature, best_value
