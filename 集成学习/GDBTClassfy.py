import numpy


def split_data(data_set, label_set, feature, point):
    dl = data_set[numpy.nonzero(data_set[:, feature] < point)[0]]
    ll = label_set[numpy.nonzero(data_set[:, feature] < point)[0]]
    dr = data_set[numpy.nonzero(data_set[:, feature] >= point)[0]]
    lr = label_set[numpy.nonzero(data_set[:, feature] >= point)[0]]
    return dl, ll, dr, lr


class Cart:
    def __init__(self, data_set, label_set, depth):
        self.tree = self.fit(data_set, label_set, depth, 0)

    def p(self, row, tree):
        if type(tree) == numpy.float64:
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

    def fit(self, data_set, label_set, depth, cur_depth):
        if cur_depth == depth:
            return label_set.mean()

        cur_depth += 1
        min_square_err = float('inf')
        for feature in range(data_set.shape[1]):
            for point in numpy.unique(data_set[:, feature].getA()):
                dl, ll, dr, lr = split_data(data_set, label_set, feature, point)
                if 0 == ll.shape[0]: ll = numpy.zeros((1, 1))
                if 0 == lr.shape[0]: lr = numpy.zeros((1, 1))
                c1 = ll.mean()
                c2 = lr.mean()
                square_err = numpy.dot((ll - c1).T, (ll - c1)) + numpy.dot((lr - c2).T, (lr - c2))
                if square_err < min_square_err:
                    min_square_err = square_err
                    best_feature = feature
                    best_point = point
        dl, ll, dr, lr = split_data(data_set, label_set, best_feature, best_point)
        return {'feature': best_feature, \
                'point': best_point, \
                'leftTree': self.fit(dl, ll, depth, cur_depth), \
                'rightTree': self.fit(dr, lr, depth, cur_depth) \
                }


class GDBTClassfier:
    def __init__(self, tree_n=10, depth=3):
        self.tree_n = tree_n
        self.tree = []
        self.depth = depth

    def linener_sum(self, data_set, **kw):
        if "cur_tree_n" not in kw.keys():
            kw["cur_tree_n"] = self.tree_n
        m, n = data_set.shape
        result = numpy.mat(numpy.zeros((m, 1)))
        for i in range(kw["cur_tree_n"]):
            result = result + self.tree[i].pred(data_set)
        return result

    def predic_pro(self, data_set, **kw):
        if "cur_tree_n" not in kw.keys():
            kw["cur_tree_n"] = self.tree_n
        m, n = data_set.shape
        result = numpy.mat(numpy.zeros((m, 1)))
        for i in range(kw["cur_tree_n"]):
            result = result + self.tree[i].pred(data_set)
        return 1 / (1 + numpy.exp(-result))

    def predic(self, data_set, **kw):
        if "cur_tree_n" not in kw.keys():
            kw["cur_tree_n"] = self.tree_n
        m, n = data_set.shape
        result = numpy.mat(numpy.zeros((m, 1)))
        for i in range(kw["cur_tree_n"]):
            result = result + self.tree[i].pred(data_set)
        result = 1 / (1 + numpy.exp(-result))
        y = numpy.mat(numpy.ones((m, 1)))
        y[result < 0.5] = -1
        return y

    def fit(self, data_set, label_set, **kw):
        if "r" not in kw.keys():
            kw["r"] = label_set / (1 + numpy.exp(label_set * 0))
        if 'cur_tree_n' not in kw.keys():
            kw['cur_tree_n'] = 0
        if kw['cur_tree_n'] < self.tree_n:
            kw['cur_tree_n'] += 1
            self.tree.append(Cart(data_set, kw["r"], self.depth))
            m = label_set.shape[0]
            new_r = numpy.zeros((m, 1))
            for i in range(m):
                new_r[i] = label_set[i] / (
                        1 + numpy.exp(label_set[i] * self.linener_sum(data_set[i], cur_tree_n=kw['cur_tree_n'])))
            self.fit(data_set, label_set, r=new_r, cur_tree_n=kw['cur_tree_n'])


if __name__ == '__main__':
    import LoadData

    x, y = LoadData.loadDataSetClf('horseColicTraining.txt')
    clf = GDBTClassfier(50, 2)
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
    0.8996655518394648
    0.8208955223880597
    """
