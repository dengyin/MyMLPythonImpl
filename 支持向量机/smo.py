import random

from numpy import *


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(alpha, H, L):
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L
    return alpha


def funG(dataMat, labelMat, alphas, b, x):
    g = 0.0
    m, n = dataMat.shape
    for i in range(m):
        g += float(alphas[i, 0] * labelMat[i, 0] * (dataMat[i, :] * x.T))
    return g + b


def Eta(xi, xj):
    return xi * xi.T + xj * xj.T - 2 * xi * xj.T


def calW(alphas, dataMat, labelMat):
    m, n = dataMat.shape
    w = zeros((n, 1))
    for i in range(alphas.shape[0]):
        w += alphas[i, 0] * labelMat[i, 0] * dataMat[i, :].T
    return w


def smo(dataMat, labelMat, C, error, maxIter):
    m, n = dataMat.shape
    alphas = mat(zeros((m, 1)))
    b = 0
    it = 0
    while (it < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            Ei = funG(dataMat, labelMat, alphas, b, dataMat[i, :]) - float(labelMat[i, 0])
            if (labelMat[i, 0] * Ei < -error and alphas[i, 0] < C) or (
                    labelMat[i, 0] * Ei > error and alphas[i, 0] > 0):
                j = selectJrand(i, m)
                Ej = funG(dataMat, labelMat, alphas, b, dataMat[j, :]) - float(labelMat[j, 0])
                alphaIold = alphas[i, 0].copy()
                alphaJold = alphas[j, 0].copy()
                ksi = alphaIold * labelMat[i, 0] + alphaJold * labelMat[j, 0]
                if labelMat[i, 0] != labelMat[j, 0]:
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaJold + alphaIold - C)
                    H = min(C, alphaJold + alphaIold)
                if L == H: continue
                eta = Eta(dataMat[i, :], dataMat[j, :])
                if eta <= 0: continue
                alphaJnew = clipAlpha(alphaJold + labelMat[j, 0] * (Ei - Ej) / eta, H, L)
                alphaInew = (ksi - alphaJnew * labelMat[j, 0]) * labelMat[i, 0]
                alphas[i, 0] = alphaInew
                alphas[j, 0] = alphaJnew
                b1 = 0
                b2 = 0
                for n in range(m):
                    b1 -= alphas[n, 0] * labelMat[n, 0] * dataMat[n, :] * dataMat[i, :].T
                    b2 -= alphas[n, 0] * labelMat[n, 0] * dataMat[n, :] * dataMat[j, :].T
                b1 += labelMat[i, 0]
                b2 += labelMat[j, 0]
                if alphas[i, 0] > 0 and alphas[i, 0] < C:
                    b = b1
                elif alphas[j, 0] > 0 and alphas[j, 0] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
        if alphaPairsChanged == 0:
            it += 1
        else:
            it = 0
    return b, alphas
