from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from my_linear_model import MyGaussianProcessRegressor

if __name__ == '__main__':
    X, y = make_regression(n_samples=1000, n_features=11)

    gpr = GaussianProcessRegressor(alpha=1, kernel=RBF(0.1))
    gpr.fit(X, y)
    y1 = gpr.predict(X)

    my_GPR = MyGaussianProcessRegressor(sigma_e=1, kernel='rbf')
    my_GPR.fit(X, y)
    y2 = my_GPR.predict(X)

    print('end')

