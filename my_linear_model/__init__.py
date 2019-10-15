from .gaussian_discriminat_analysis import MyGaussianDiscriminatAnalysis
from .linear_regression import MyLinearRegression, MyRidge, MyLasso, MyGaussianProcessRegression, \
    MyBayesianLinearRegression
from .logistic_regression import MyLogisticRegression, MyBayesianLogisticRegression, MyGaussianProcessClassifier, \
    MyKernelLogisticRegression
from .softmax_regression import MySoftmaxRegression

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
    'MySoftmaxRegression',
    'MyKernelLogisticRegression'
]
