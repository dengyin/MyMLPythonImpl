import numpy as np
from sklearn.datasets import make_classification


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


class NeuralNetworkClassfier:
    def __init__(self, hidden_layer_sizes=(100,), activation="sigmod", learning_rate=0.1, max_iter=100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        self.w = []
        self.b = []
        hidden_layer_sizes = []
        for l in range(len(self.hidden_layer_sizes) + 2):
            if l is 0:
                hidden_layer_sizes.append(X.shape[1])
            elif l is len(self.hidden_layer_sizes) + 1:
                hidden_layer_sizes.append(1)
            else:
                hidden_layer_sizes.append(self.hidden_layer_sizes[l - 1])
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)

        # 初始化参数
        for l in range(len(self.hidden_layer_sizes)):
            if l is 0:
                self.w.append(np.ones((self.hidden_layer_sizes[l], X.shape[1])))
                self.b.append(np.zeros((self.hidden_layer_sizes[l], 1)))
            else:
                self.w.append(np.random.random((self.hidden_layer_sizes[l], self.hidden_layer_sizes[l - 1])))
                self.b.append(np.zeros((self.hidden_layer_sizes[l], 1)))

        for i in range(self.max_iter):
            self.BP(X, y)

    def BP(self, X, y):
        z = [i for i in range(len(self.hidden_layer_sizes))]
        a = [i for i in range(len(self.hidden_layer_sizes))]
        for l in range(len(self.hidden_layer_sizes)):
            if l is 0:
                z[l] = X.T
                a[l] = z[l]
            else:
                z[l] = self.w[l] * a[l - 1] + self.b[l]
                a[l] = sigmoid(z[l])
        for l in range(len(self.hidden_layer_sizes))[::-1]:
            if l is len(self.hidden_layer_sizes) - 1:
                dz = a[l] - y.T
            elif l is 0:
                continue
            else:
                dz = np.array(da) * np.array(sigmoid(z[l])) * np.array(1 - sigmoid(z[l]))
            dw = np.dot(dz, a[l - 1].T) / X.shape[0]
            db = np.mean(np.mat(dz).getA(), axis=1, keepdims=True)
            da = np.dot(self.w[l].T, dz)
            self.w[l] = self.w[l] - self.learning_rate * dw
            self.b[l] = self.b[l] - self.learning_rate * db

    def predict_pro(self, X):
        t = X.T
        for l in range(1, len(self.hidden_layer_sizes)):
            z = self.w[l] * t + self.b[l]
            t = sigmoid(z)
        return t.T

    def predict(self, X):
        t = self.predict_pro(X)
        t[t >= 0.5] = 1
        t[t < 0.5] = 0
        return t


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=10, n_redundant=2)
    clf = NeuralNetworkClassfier(hidden_layer_sizes=(1,), learning_rate=0.01, max_iter=1000)
    clf.fit(X, y)
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    y_scores = clf.predict_pro(X)
    print("train roc_auc:", roc_auc_score(y.getA(), y_scores))
    y_pred = clf.predict(X)
    print("train f1:", f1_score(y.getA(), y_pred))
    print("train accu:", accuracy_score(y.getA(), y_pred))

    X_test, y_test = LoadData.loadDataSet('horseColicTest.txt')
    y_scores = clf.predict_pro(X_test)
    print("test roc_auc:", roc_auc_score(y_test.getA(), y_scores))
    y_pred = clf.predict(X_test)
    print("test f1:", f1_score(y_test.getA(), y_pred))
    print("test accu:", accuracy_score(y_test.getA(), y_pred))

    print('-----')
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(activation="relu", hidden_layer_sizes=(1,), random_state=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("train f1:", f1_score(y.getA(), y_pred))
    print("train accu:", accuracy_score(y.getA(), y_pred))

    X_test, y_test = LoadData.loadDataSet('horseColicTest.txt')
    y_pred = clf.predict(X_test)
    print("test f1:", f1_score(y_test.getA(), y_pred))
    print("test accu:", accuracy_score(y_test.getA(), y_pred))
