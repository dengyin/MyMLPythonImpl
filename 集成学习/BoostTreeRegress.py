import numpy


class Cart:

    def pred(self, data_set, feature, point, c1, c2):
        m = data_set.shape[0]
        result = numpy.mat(numpy.zeros((m, 1)))
        for row in range(m):
            if data_set[row, feature] > point:
                result[row] = c1
            else:
                result[row] = c2
        return result

    def fit(self, data_set, label_set):
        m, n = data_set.shape
        min_e = float('inf')
        self.feature = -1.0
        self.point = float('inf')
        for feature in range(n):
            for point in numpy.unique(data_set[:, feature].getA()):
                c1 = label_set[data_set[:, feature] >= point].mean()
                c2 = label_set[data_set[:, feature] < point].mean()
                pred_label = self.pred(data_set, feature, point, c1, c2)
                e = (label_set - pred_label).T * (label_set - pred_label)
                if e < min_e:
                    min_e = e
                    self.feature = feature
                    self.point = point
                    self.c1 = c1
                    self.c2 = c2


class BoostTreeRegresser:
    def __init__(self, weak_regresser_num=100):
        self.weak_regresser = []
        self.weak_regresser_num = weak_regresser_num

    def predic(self, data_set):
        m = data_set.shape[0]
        result = numpy.mat(numpy.zeros((m, 1)))
        for t in range(self.weak_regresser_num):
            result = result + self.weak_regresser[t].pred(data_set, self.weak_regresser[t].feature, \
                                                          self.weak_regresser[t].point, self.weak_regresser[t].c1, \
                                                          self.weak_regresser[t].c2)
        return result

    def fit(self, data_set, label_set):
        r = []
        r.append(label_set)
        for t in range(self.weak_regresser_num):
            self.weak_regresser.append(Cart())
            self.weak_regresser[t].fit(data_set, r[t])
            predic_r = numpy.mat(numpy.zeros((data_set.shape[0], 1)))
            for i in range(t + 1):
                predic_r = predic_r + self.weak_regresser[i].pred(data_set, self.weak_regresser[i].feature, \
                                                                  self.weak_regresser[i].point, \
                                                                  self.weak_regresser[i].c1, \
                                                                  self.weak_regresser[i].c2)
            print(t, " feature、point、c1、c2:", self.weak_regresser[i].feature, self.weak_regresser[i].point,
                  self.weak_regresser[i].c1, self.weak_regresser[i].c2)
            r.append(label_set - predic_r)


if __name__ == '__main__':
    import LoadData

    x, y = LoadData.loadDataSetReg('abalone.txt')
    reg = BoostTreeRegresser(20)
    reg.fit(x, y)
    reg_y = reg.predic(x)
    r_2 = (y - reg_y).T * (y - reg_y)
    m = y.shape[0]
    result = r_2[0, 0] ** 0.5 / m
    print('r2', r_2)
    print('m', m)
    print(result)

    """
    r2 [[ 21424.75884606]]
    m 4177
    0.0350423720169
    """
