from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, accuracy_score

from my_linear_model import MyLogisticRegression, MyGaussianDiscriminatAnalysis

if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=4)

    clf = MyLogisticRegression(steps=0.005, normalize='l2', alpha=1, solver='gradient_descent')
    clf.fit(X, y)
    print(clf.coef_)
    print(log_loss(y, clf.predict_proba(X)))
    print(accuracy_score(y, clf.predict(X)))
    print('MyLogisticRegression===============gradient_descent')

    clf = MyLogisticRegression(steps=0.05, normalize='l2', alpha=1, solver='newton', stop_threshold=1e-8)
    clf.fit(X, y)
    print(clf.coef_)
    print(log_loss(y, clf.predict_proba(X)))
    print(accuracy_score(y, clf.predict(X)))
    print('MyLogisticRegression===============newton')

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(penalty='l2', C=1.0)
    clf.fit(X, y)
    print(clf.coef_)
    print(log_loss(y, clf.predict_proba(X)))
    print(accuracy_score(y, clf.predict(X)))
    print('LogisticRegression===============')

    clf = MyGaussianDiscriminatAnalysis()
    clf.fit(X, y)
    print(clf.coef_)
    print(log_loss(y, clf.predict_proba(X)))
    print(accuracy_score(y, clf.predict(X)))
    print('MyGaussianDiscriminatAnalysis===============')

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    print(clf.coef_)
    print(log_loss(y, clf.predict_proba(X)))
    print(accuracy_score(y, clf.predict(X)))
    print('LinearDiscriminantAnalysis===============')
