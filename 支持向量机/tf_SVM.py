import numpy as np
import tensorflow as tf


class SVM():
    def __init__(self, rate=0.01, iter=1000, C=0.01):
        self.sess = tf.Session()
        self.rate = rate
        self.iter = iter
        self.C = C

    def predict(self, X):
        result = np.sign(np.matmul(X, self.w) + self.b)
        result[result == -1] = 0
        return result

    def fit(self, X, y):
        m, n = X.shape
        input = tf.placeholder(dtype=tf.float64, shape=(None, n))
        output = tf.placeholder(dtype=tf.float64, shape=(None, 1))
        w = tf.Variable(initial_value=tf.zeros(shape=(n, 1), dtype=tf.float64), dtype=tf.float64)
        b = tf.Variable(initial_value=0., dtype=tf.float64)
        loss = tf.reduce_mean(tf.maximum(tf.constant(0, dtype=tf.float64),
                                         1 - output * (tf.matmul(input, w) + b))) + self.C * tf.reduce_mean(
            tf.square(w))
        optimizer = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(self.iter):
            self.sess.run(optimizer, feed_dict={input: X, output: y.reshape((-1, 1))})
            # print('loss:'+str(self.sess.run(loss,feed_dict={input: X, output: y.reshape((-1,1))})))
        self.w = self.sess.run(w.value())
        self.b = self.sess.run(b.value())


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC

    X, y = make_classification(100, 5)

    classfy = SVM(rate=0.01, iter=5000, C=0.0001)
    classfy.fit(X, y)
    y_hat = classfy.predict(X)
    print(f1_score(y, y_hat))

    classfy = SVC(C=0.0001, kernel='linear')
    classfy.fit(X, y)
    y_hat = classfy.predict(X)
    print(f1_score(y, y_hat))
