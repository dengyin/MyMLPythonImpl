# coding=utf-8

from numpy import *


def combineMat(x, y):
    return np.hstack((x, y))


def divideMat(matrix):
    return matrix[:, 0:-1], matrix[:, -1]


def binSplitDataSet(dataSet, feat, point):
    R1 = R2 = None
    for dataIndex in range(dataSet.shape[0]):
        if dataSet[dataIndex, feat] < point:
            if type(R1) == type(None):
                R1 = dataSet[dataIndex]
            else:
                R1 = np.vstack((R1, dataSet[dataIndex]))
        else:
            if type(R2) == type(None):
                R2 = dataSet[dataIndex]
            else:
                R2 = np.vstack((R2, dataSet[dataIndex]))

    return R1, R2


def getC1C2(dataSet, feat, splitPoint):
    c1Sum = c2Sum = 0.0
    c1Num = c2Num = 0
    for dataIndex in range(dataSet.shape[0]):
        if dataSet[dataIndex, feat] < splitPoint:
            c1Num += 1
            c1Sum += dataSet[dataIndex, -1]
        else:
            c2Num += 1
            c2Sum += dataSet[dataIndex, -1]
    if c1Num == 0:
        c1 = 0.0
    else:
        c1 = c1Sum / c1Num
    if c2Num == 0:
        c2 = 0.0
    else:
        c2 = c2Sum / c2Num
    return c1, c2


def calRSS(dataSet, feat, point, c1, c2):
    rss = 0.0
    for dataIndex in range(dataSet.shape[0]):
        if dataSet[dataIndex, feat] < point:
            rss += (dataSet[dataIndex, -1] - c1) ** 2
        else:
            rss += (dataSet[dataIndex, -1] - c2) ** 2
    return rss


def findBestPoint(dataSet, feat):
    bestRSS = splitPoint = float("inf")
    for dataIndex in range(dataSet.shape[0]):
        point = dataSet[dataIndex, feat]
        c1, c2 = getC1C2(dataSet, feat, point)
        curRSS = calRSS(dataSet, feat, point, c1, c2)
        if curRSS < bestRSS:
            bestRSS = curRSS
            splitPoint = point
    return splitPoint


def findBestFeat_Point(dataSet, minRSSError=1, minDataNum=5):
    bestFeat = -1
    bestRSS = splitPoint = float("inf")
    for curFeat in range(dataSet.shape[1] - 1):
        point = findBestPoint(dataSet, curFeat)
        c1, c2 = getC1C2(dataSet, curFeat, point)
        curRSS = calRSS(dataSet, curFeat, point, c1, c2)
        if curRSS < bestRSS:
            bestRSS = curRSS
            bestFeat = curFeat
            splitPoint = point
    if np.var(dataSet[:, -1]) * dataSet.shape[0] - bestRSS < minRSSError or dataSet.shape[0] <= minDataNum:
        bestFeat = -1
        splitPoint = np.mean(dataSet[:, -1])
    return bestFeat, splitPoint


def creatTree(dataSet, minRSSError=1, minDataNum=5):
    feat, point = findBestFeat_Point(dataSet, minRSSError, minDataNum)
    if feat == -1: return point
    tree = {}
    tree['spInd'] = feat
    tree['point'] = point
    lTree, rTree = binSplitDataSet(dataSet, feat, point)
    tree['left'] = creatTree(lTree)
    tree['right'] = creatTree(rTree)
    return tree
