import math
import random

import numpy


def cal_gini(label_set):
    m = label_set.shape[1]
    gini = 0.0
    for c in numpy.unique(label_set.getA()):
        gini = gini + (label_set[label_set == c].shape[1] / m) ** 2
    gini = 1 - gini
    return gini


class Cart:
    def __init__(self, data_set, label_set, depth):
        m, n = data_set.shape
        self.features = random.sample(range(0, n), math.ceil(n ** 0.5))
        self.tree = self.fit(data_set, label_set, depth, 0)

    def p(self, row, tree):
        if tree == 1 or tree == -1:
            return tree
        else:
            if row[0, tree['feature']] >= tree['point']:
                return self.p(row, tree['rightTree'])
            else:
                return self.p(row, tree['leftTree'])

    def pred(self, data_set):
        m, n = data_set.shape
        result = numpy.mat(numpy.ones((m, 1)))
        for i in range(m):
            result[i] = self.p(data_set[i], self.tree)
        return result

    def fit(self, data_set, label_set, depth, cur_dep=0):
        if cur_dep == depth:
            if label_set.mean() > 0:
                return 1
            else:
                return -1

        cur_dep += 1
        m, n = data_set.shape
        min_gain_gini = float('inf')
        for feature in self.features:
            for point in numpy.unique(data_set[:, feature].getA()):
                gini_point = label_set[data_set[:, feature] >= point].shape[1] / m * \
                             cal_gini(label_set[data_set[:, feature] >= point]) + \
                             label_set[data_set[:, feature] < point].shape[1] / m * \
                             cal_gini(label_set[data_set[:, feature] < point])

                if gini_point < min_gain_gini:
                    min_gain_gini = gini_point
                    best_feature = feature
                    best_point = point
        return {'feature': best_feature, \
                'point': best_point, \
                'leftTree': self.fit(data_set[numpy.nonzero(data_set[:, best_feature] < best_point)[0]],
                                     label_set[numpy.nonzero(data_set[:, best_feature] < best_point)[0]], depth,
                                     cur_dep), \
                'rightTree': self.fit(data_set[numpy.nonzero(data_set[:, best_feature] >= best_point)[0]],
                                      label_set[numpy.nonzero(data_set[:, best_feature] >= best_point)[0]], depth,
                                      cur_dep) \
                }


class RFClassfier:
    def __init__(self, tree_n=10, depth=3):
        self.tree_n = tree_n
        self.tree = []
        self.depth = depth

    def predic_pro(self, data_set):
        m, n = data_set.shape
        result = numpy.mat(numpy.zeros((m, 1)))
        for i in range(self.tree_n):
            result = result + self.tree[i].pred(data_set)
        result = result / self.tree_n
        return result

    def predic(self, data_set):
        m, n = data_set.shape
        result = numpy.mat(numpy.zeros((m, 1)))
        for i in range(self.tree_n):
            result = result + self.tree[i].pred(data_set)
        result = result / self.tree_n
        result[result >= 0] = 1
        result[result < 0] = -1
        return result

    def fit(self, data_set, label_set):
        m, n = data_set.shape
        for i in range(self.tree_n):
            li = [random.randint(0, m - 1) for p in range(m)]
            self.tree.append(Cart(data_set[li], label_set[li], self.depth))


if __name__ == '__main__':
    import LoadData

    x, y = LoadData.loadDataSetClf('horseColicTraining.txt')

    clf = RFClassfier(100, 3)
    clf.fit(x, y)
    p = clf.predic(x)
    m = 0
    for i in range(y.shape[0]):
        if y[i] == p[i]:
            m += 1.0
    result = m / y.shape[0]
    print(result)

    _x, _y = LoadData.loadDataSetClf('horseColicTest.txt')
    py = clf.predic(_x)
    m = 0
    for i in range(_y.shape[0]):
        if _y[i] == py[i]:
            m += 1.0
    result = m / _y.shape[0]
    print(result)

    """
    0.8060200668896321
    0.8059701492537313
    """
