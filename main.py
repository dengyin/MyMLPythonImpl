from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA

from my_decomposition import MyFactorAnalysis, MyPCA, MyKernelPCA

if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=10, n_redundant=2)

    my_fa = MyFactorAnalysis(n_components=4)
    z1 = my_fa.fit_transform(X)

    fa = FactorAnalysis(n_components=4)
    z2 = fa.fit_transform(X)

    my_pca = MyPCA(n_components=4)
    z3 = my_pca.fit_transform(X)

    pca = PCA(n_components=4)
    z4 = pca.fit_transform(X)

    my_kpca = MyKernelPCA(n_components=4)
    z5 = my_kpca.fit_transform(X)

    kpca = KernelPCA(n_components=4)
    z6 = kpca.fit_transform(X)

    print('end')
