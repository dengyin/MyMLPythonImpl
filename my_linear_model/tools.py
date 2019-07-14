import numpy as np


def sigma(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(y_true, y_pred):
    return -np.sum(np.log(y_pred) * y_true)


def iter_optimization(
        X,
        y,
        step_func,
        score_func,
        predic_func,
        steps,
        stop_threshold):
    w = np.random.random((X.shape[1], 1))
    l_of_w_first = score_func(y, predic_func(X, w))
    while True:
        step = step_func(X, y, w)
        w = w - steps * step
        l_of_w_last = score_func(y, predic_func(X, w))
        if l_of_w_first - l_of_w_last < stop_threshold:
            break
        l_of_w_first = l_of_w_last
    return w
