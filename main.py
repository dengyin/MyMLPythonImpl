from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

from my_decomposition import MyFactorAnalysis
from my_linear_model import MyLogisticRegression

if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=10)
    # mp = FactorAnalysis()
    # mp.fit(X)
    # Z1 = mp.transform(X)

    my_fa = MyFactorAnalysis()
    Z2 = my_fa.fit_transform(X)
    t = my_fa.scalings_.T @ my_fa.scalings_
    clf = MyLogisticRegression(solver='newton')
    clf.fit(Z2, y)
    print(log_loss(y, clf.predict_proba(Z2)))
