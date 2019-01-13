from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from my_linear_model import MySoftmaxRegression

if __name__ == '__main__':
    X, y = make_classification(n_classes=3, n_informative=18)
    my_clf = MySoftmaxRegression(steps=0.0001, stop_threshold=1e-6, normalize='l2', alpha=1)
    my_clf.fit(X, y)
    y1 = my_clf.predict_proba(X)
    y2 = my_clf.predict(X).ravel()
    print(accuracy_score(y, y2))

    clf = LogisticRegression(multi_class='auto')
    clf.fit(X, y)
    y3 = clf.predict_proba(X)
    y4 = clf.predict(X)
    print(accuracy_score(y, y4))
    print('end')

