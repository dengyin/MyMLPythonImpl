from sklearn.datasets import make_classification

from my_decomposition import MyPCA

if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=4)

    from sklearn.decomposition.pca import PCA

    p = PCA()
    p.fit(X)
    Z1 = p.transform(X)

    mp = MyPCA()
    mp.fit(X)
    Z2 = mp.transform(X)

    print(Z1 - Z2)
