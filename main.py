import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.metrics import log_loss, accuracy_score

from my_linear_model import MyLogisticRegression, MyGaussianProcessClassifier

if __name__ == '__main__':
    X, y = make_circles()

    clf = MyLogisticRegression()
    clf.fit(X, y)
    y1 = clf.predict_proba(X)
    print(log_loss(y, y1))
    print(accuracy_score(y, clf.predict(X)))

    my_clf = MyGaussianProcessClassifier(kernel='rbf', kernel_para=1, alpha=1)
    my_clf.fit(X, y)
    y2 = my_clf.predict_proba(X)
    print(log_loss(y, y2))
    print(accuracy_score(y, my_clf.predict(X)))

    x0, x1 = np.meshgrid(np.linspace(X[:, 0].min() * 1.1, X[:, 0].max() * 1.1, 300).reshape((-1, 1)),
                         np.linspace(X[:, 1].min() * 1.1, X[:, 1].max() * 1.1, 300).reshape((-1, 1)))

    X_new = np.hstack((x0.ravel().reshape((-1, 1)), x1.ravel().reshape((-1, 1))))

    y_predict = my_clf.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(X[:, 0].min() * 1.1, X[:, 0].max() * 1.1)
    plt.ylim(X[:, 1].min() * 1.1, X[:, 1].max() * 1.1)

    plt.show()

    print('end')

