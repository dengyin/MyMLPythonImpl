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
                c1 = ll.mean()
                c2 = lr.mean()
                square_err = (ll - c1).T * (ll - c1) + (lr - c2).T * (lr - c2)
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


class GDBTRegressor:
    def __init__(self, tree_n=10, depth=3):
        self.tree_n = tree_n
        self.tree = []
        self.depth = depth

    def predic(self, data_set, **kw):
        if "cur_tree_n" not in kw.keys():
            kw["cur_tree_n"] = self.tree_n
        m, n = data_set.shape
        result = numpy.mat(numpy.zeros((m, 1)))
        for i in range(kw["cur_tree_n"]):
            result = result + self.tree[i].pred(data_set)
        return result

    def fit(self, data_set, label_set, **kw):
        if "r" not in kw.keys():
            kw["r"] = label_set
        if 'cur_tree_n' not in kw.keys():
            kw['cur_tree_n'] = 0
        if kw['cur_tree_n'] < self.tree_n:
            kw['cur_tree_n'] += 1
            self.tree.append(Cart(data_set, kw["r"], self.depth))
            new_r = 2 * (label_set - self.predic(data_set, cur_tree_n=kw['cur_tree_n']))
            self.fit(data_set, label_set, r=new_r, cur_tree_n=kw['cur_tree_n'])


if __name__ == '__main__':
    import LoadData

    x, y = LoadData.loadDataSetReg('abalone.txt')
    reg = GDBTRegressor(25, 4)
    reg.fit(x, y)

    reg_y = reg.predic(x)
    r_2 = (y - reg_y).T * (y - reg_y)
    m = y.shape[0]
    result = r_2[0, 0] ** 0.5 / m
    print('r2', r_2)
    print('m', m)
    print(result)

    """
    r2 [[ 21892.98203715]]
    m 4177
    0.0354232158177
    """
