import numpy


class Cart:
    def __init__(self):
        pass

    def fit(self, data_set, label_set, weight):
        m, n = data_set.shape
        min_e = float('inf')
        for feature in range(n):
            range_min = data_set[:, feature].min()
            range_max = data_set[:, feature].max()
            step = (range_max - range_min) / 100.0
            for row in range(m):
                for thresh_ineq in ['lt', 'gt']:
                    thresh_val = range_min + row * step
                    predic_y = self.pred(data_set, thresh_val, thresh_ineq, feature)
                    err = numpy.mat(numpy.ones((m, 1)))
                    err[predic_y == label_set] = 0
                    err = err.T * weight
                    if err < min_e:
                        min_e = err
                        self.thresh_ineq = thresh_ineq
                        self.thresh_val = thresh_val
                        self.feature = feature
        return min_e

    def pred(self, x, thresh_val, thresh_ineq, feature):
        result = numpy.mat(numpy.ones((x.shape[0], 1)))
        if thresh_ineq == 'lt':
            result[x[:, feature] > thresh_val] = -1
        else:
            result[x[:, feature] <= thresh_val] = -1
        return result


class BoostTreeClassfier:
    def __init__(self, weak_classfier_num=100):
        self.aplhas = []
        self.weak_classfier = []
        self.weak_classfier_num = weak_classfier_num

    def predic(self, data_set):
        m, n = data_set.shape
        result = numpy.mat(numpy.ones((m, 1)))
        y = numpy.mat(numpy.zeros((m, 1)))
        for row in range(m):
            for t in range(self.weak_classfier_num):
                y[row] += self.aplhas[t] * self.weak_classfier[t].pred(data_set[row, :],
                                                                       self.weak_classfier[t].thresh_val,
                                                                       self.weak_classfier[t].thresh_ineq,
                                                                       self.weak_classfier[t].feature)
        result[y < 0] = -1
        return result

    def predic_pro(self, data_set):
        m, n = data_set.shape
        y = numpy.mat(numpy.zeros((m, 1)))
        for row in range(m):
            for t in range(self.weak_classfier_num):
                y[row] += self.aplhas[t] * self.weak_classfier[t].pred(data_set[row, :],
                                                                       self.weak_classfier[t].thresh_val,
                                                                       self.weak_classfier[t].thresh_ineq,
                                                                       self.weak_classfier[t].feature)
        return y

    def fit(self, data_set, label_set):
        m, n = data_set.shape
        d = []
        e = []
        d.append(numpy.mat(numpy.ones((m, 1))) / float(m))
        for t in range(self.weak_classfier_num):
            self.weak_classfier.append(Cart())
            e.append(self.weak_classfier[t].fit(data_set, label_set, d[t]))
            self.aplhas.append(0.5 * numpy.log((1 - e[t]) / max(e[t], 1e-16)))
            d.append(numpy.mat(numpy.ones((m, 1))) / float(m))
            for i in range(m):
                d[t + 1][i] = d[t][i] * numpy.exp(
                    -1.0 * self.weak_classfier[t].pred(data_set[i], self.weak_classfier[t].thresh_val,
                                                       self.weak_classfier[t].thresh_ineq,
                                                       self.weak_classfier[t].feature) * label_set[i] * self.aplhas[
                        t]) / d[t].sum()


if __name__ == '__main__':
    import LoadData

    x, y = LoadData.loadDataSetClf('horseColicTraining.txt')

    clf = BoostTreeClassfier(100)
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
    0.8394648829431438
    0.7761194029850746
    """
