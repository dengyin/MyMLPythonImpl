import numpy as np


class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def __calc_shannon_ent(self, y):
        ent = 0.0
        p = {}
        for k in np.unique(y):
            p[k] = len(y[y == k]) / len(y)
            ent += (-p[k] * np.log(p[k]))
        return ent

    def __calc_ent_gain(self, X, y, feature):
        m, n = X.shape
        ent_gain = self.__calc_shannon_ent(y)
        points = np.unique(X[:, feature])
        for point in points:
            split_y = y[X[:, feature] == point]
            ent_gain -= (len(split_y) / m) * self.__calc_shannon_ent(split_y)
        return ent_gain

    def __calc_ent_gain_ratio(self, X, y, is_discrete, feature, point):
        m, n = X.shape
        ent_gain = self.__calc_shannon_ent(y)
        ent_feature = 0
        if is_discrete:
            for p in np.unique(X[:, feature]):
                split_y = y[X[:, feature] == p]
                ent_gain -= (len(split_y) / m) * self.__calc_shannon_ent(split_y)
                ent_feature += (-len(split_y) / m) * np.log(len(split_y) / m)
        else:
            y1 = y[X[:, feature] <= point]
            y2 = y[X[:, feature] > point]
            ent_gain -= (len(y1) / m) * self.__calc_shannon_ent(y1) + (len(y2) / m) * self.__calc_shannon_ent(y2)
            ent_feature = (-len(y1) / m) * np.log(len(y1) / m) + (-len(y2) / m) * np.log(len(y2) / m)

        return ent_gain / ent_feature

    def __calc_gini(self, y):
        gini = 1
        p = {}
        for k in np.unique(y):
            p[k] = len(y[y == k]) / len(y)
            gini -= p[k] ** 2
        return gini

    def __calc_gini_gain(self, X, y, feature, is_discrete, point):
        m, n = X.shape
        gini_gain = self.__calc_gini(y)
        if is_discrete:
            y1 = y[X[:, feature] == point]
            y2 = y[X[:, feature] != point]
        else:
            y1 = y[X[:, feature] <= point]
            y2 = y[X[:, feature] > point]
        gini_gain = gini_gain - (len(y1) / m) * self.__calc_gini(y1) - (len(y2) / m) * self.__calc_gini(y2)
        return gini_gain

    def __find_best_split(self, X, y):
        max_gain = float('-inf')
        best_feature = best_point = None
        for feature in self.__features:
            # 判断该特征是否为离散特征
            # 1.全是离散值
            # 2.最大值==特征数量-1
            is_discrete = \
                np.sum((X[:, feature] // 1) - X[:, feature]) == 0 \
                and np.max(X[:, feature]) == len(np.unique(X[:, feature])) - 1
            # ID3处理离散特征
            if self.criterion == 'ID3' and is_discrete:
                cur_gain = self.__calc_ent_gain(X, y, feature)
                if cur_gain > max_gain:
                    best_feature = feature
            # C4.5处理离散特征
            elif self.criterion == 'C4.5' and is_discrete:
                cur_gain = self.__calc_ent_gain_ratio(X, y, feature, is_discrete, None)
                if cur_gain > max_gain:
                    best_feature = feature
            # C4.5处理连续特征
            elif self.criterion == 'C4.5' and not is_discrete:
                for point in np.unique(X[:, feature]):
                    cur_gain = self.__calc_ent_gain_ratio(X, y, feature, is_discrete, point)
                    if cur_gain > max_gain:
                        best_feature = feature
                        best_point = point
            # Gini
            elif self.criterion == 'gini':
                for point in np.unique(X[:, feature]):
                    cur_gain = self.__calc_gini_gain(X, y, feature, is_discrete, point)
                    if cur_gain > max_gain:
                        best_feature = feature
                        best_point = point
        self.__features.remove(best_feature)
        return best_feature, best_point, is_discrete

    def __get_proba_of_dataset(self, y):
        result = []
        for c in self.classes_:
            p = len(y[y == c]) / len(y)
            result.append(p)
        return result

    def __creat_tree(self, X, y, cur_deep):
        # 达到最大深度或样本小于等于min_samples_leaf时停止分裂
        if (cur_deep is not None and cur_deep >= self.max_depth) or len(y) <= self.min_samples_leaf:
            return self.__get_proba_of_dataset(y)

        cur_deep += 1
        temp_dic = {}
        best_feature, best_point, is_discrete = self.__find_best_split(X, y)

        # ID3 & C4.5 离散特征
        if (self.criterion == 'ID3' or self.criterion == 'C4.5') and is_discrete:
            # todo
            pass
        # elif
        # Gini
        else:
            left_X, left_y, right_X, right_y = self.__split_by_feature_and_point(X, y, best_feature, best_point)
            temp_dic['feature'] = best_feature
            temp_dic['point'] = best_point
            # 分裂后样本小于min_samples_split则停止分裂
            if len(left_y) <= self.min_samples_split:
                temp_dic['left'] = self.__get_proba_of_dataset(left_y)
            else:
                temp_dic['left'] = self.__creat_tree(left_X, left_y, cur_deep)
            if len(right_y) <= self.min_samples_split:
                temp_dic['right'] = self.__get_proba_of_dataset(right_y)
            else:
                temp_dic['right'] = self.__creat_tree(right_X, right_y, cur_deep)

        return temp_dic

    def fit(self, X, y):
        init_cur = 1
        self.classes_ = np.unique(y)
        self.__features = set([i for i in range(X.shape[0])])
        if self.max_depth is None:
            init_cur = None
        self.dic_tree = self.__creat_tree(X, y, init_cur)
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


class DecisionTreeRegressor:
    pass
