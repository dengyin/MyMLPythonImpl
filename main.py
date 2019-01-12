from sklearn.datasets import make_classification

from my_linear_model import MyLogisticRegression, MyBayesianLogisticRegression

if __name__ == '__main__':
    X, y = make_classification(n_features=10)
    my_lr = MyLogisticRegression(alpha=0.1, normalize='l2')
    my_lr.fit(X, y)
    y1 = my_lr.predict_proba(X)

    my_blr = MyBayesianLogisticRegression(alpha=5, predict_way='expect')
    my_blr.fit(X, y)
    y2 = my_blr.predict_proba(X)

    my_blr2 = MyBayesianLogisticRegression(alpha=5, predict_way='monte_carlo')
    my_blr2.fit(X, y)
    y3 = my_blr2.predict_proba(X)

    print('end')

