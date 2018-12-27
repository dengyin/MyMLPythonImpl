from numpy import *


def calGini(dataSet):
    # if type(dataSet)==type(None):return 0.0
    D = {}
    Gini = 0.0
    for dataIndex in range(dataSet.shape[0]):
        if dataSet[dataIndex, -1] not in D.keys():
            D[dataSet[dataIndex, -1]] = 0
        D[dataSet[dataIndex, -1]] += 1
    for key in D.keys():
        Gini += (D[key] / dataSet.shape[0]) ** 2
    Gini = 1 - Gini
    return Gini


def binSplitDataSet(dataSet, feat, value):
    D1 = D2 = None
    for dataIndex in range(dataSet.shape[0]):
        if dataSet[dataIndex, feat] == value:
            if type(D1) == type(None):
                D1 = dataSet[dataIndex]
            else:
                D1 = np.vstack((D1, dataSet[dataIndex]))
        else:
            if type(D2) == type(None):
                D2 = dataSet[dataIndex]
            else:
                D2 = np.vstack((D2, dataSet[dataIndex]))
    return D1, D2


def allValueinFeat(dataSet, feat):
    return set(dataSet[:, feat])


def findBestValue(dataSet, feat):
    dataNum = float(dataSet.shape[0])
    bestGini = float('inf')
    for value in dataSet[:, feat]:
        D1, D2 = binSplitDataSet(dataSet, feat, value[0, 0])
        if type(D2) == type(None):
            Gini = calGini(D1) * D1.shape[0] / dataNum
        elif type(D1) == type(None):
            Gini = calGini(D2) * D2.shape[0] / dataNum
        else:
            Gini = calGini(D1) * D1.shape[0] / dataNum + calGini(D2) * D2.shape[0] / dataNum
        if Gini < bestGini:
            bestGini = Gini
            returnValue = value[0, 0]
    return returnValue, bestGini


def finBestFeatAndValue(dataSet):
    bestGini = float('inf')
    for featIndex in range(dataSet.shape[1] - 1):
        value, Gini = findBestValue(dataSet, featIndex)
        if Gini < bestGini:
            bestGini = Gini
            returnFeatIndex = featIndex
            returnValue = value
    return returnFeatIndex, returnValue


def mostClass(dataSet):
    countValue = {}
    for value in dataSet[:, -1]:
        if value[0, 0] not in countValue.keys():
            countValue[value[0, 0]] = 0
        countValue[value[0, 0]] += 1
    mostNum = 0
    for value in countValue.keys():
        if countValue[value] > mostNum:
            mostNum = countValue[value]
            returnValue = value
    return returnValue


def creatTree(dataSet, label, stopGini, minNum):
    if calGini(dataSet) < stopGini or dataSet.shape <= minNum:
        return mostClass(dataSet)
    featIndex, value = finBestFeatAndValue(dataSet)
    # if featIndex==-1:return value
    tree = {}
    tree['spInd'] = label[featIndex]
    tree['point'] = value
    lTree, rTree = binSplitDataSet(dataSet, featIndex, value)
    if type(lTree) != type(None):
        tree['left'] = creatTree(lTree, label, stopGini, minNum)
    if type(rTree) != type(None):
        tree['right'] = creatTree(rTree, label, stopGini, minNum)
    return tree
