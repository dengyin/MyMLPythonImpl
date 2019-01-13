from .my_gaussian_discriminat_analysis import MyGaussianDiscriminatAnalysis
from .my_linear_regression import MyLinearRegression, MyRidge, MyLasso, MyGaussianProcessRegression, \
    MyBayesianLinearRegression
from .my_logistic_regression import MyLogisticRegression, MyBayesianLogisticRegression, MyGaussianProcessClassifier
from .my_softmax_regression import MySoftmaxRegression

__all__ = [
    'MyLinearRegression',
    'MyRidge',
    'MyLasso',
    'MyGaussianProcessRegression',
    'MyLogisticRegression',
    'MyGaussianDiscriminatAnalysis',
    'MyBayesianLinearRegression',
    'MyBayesianLogisticRegression',
    'MyGaussianProcessClassifier',
    'MySoftmaxRegression'
]
