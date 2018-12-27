# coding=utf-8
from numpy import *


def calEntropy(classSet):
    dataNum = classSet.shape[0]
    ent = 0.0
    classCount = {}
    for c in classSet.tolist():
        if c[0] not in classCount.keys():
            classCount[c[0]] = 0
        classCount[c[0]] += 1
    for key in classCount:
        prob = float(classCount[key]) / dataNum
        ent = ent - prob * math.log(prob, 2)
    return ent


def datasAfterSplitByFeature(dataSet, classSet, featureIndex):
    m = hstack((dataSet, classSet))
    datas = {}
    for i in range(m.shape[0]):
        if m[i, featureIndex] not in datas.keys():
            datas[m[i, featureIndex]] = m[i]
        else:
            datas[m[i, featureIndex]] = vstack((datas[m[i, featureIndex]], m[i]))

    return datas


def deleteFeature(dataSet, classSet, classLabel, featureIndex):
    featNum = dataSet.shape[1]
    dataSet = datasAfterSplitByFeature(dataSet, classSet, featureIndex)
    # 删除classLabel对应的特征

    if featureIndex != 0 and featureIndex != featNum:
        classLabel = hstack((classLabel[0:featureIndex], classLabel[featureIndex + 1:]))
    elif featureIndex == 0:
        classLabel = classLabel[1:]
    elif featureIndex == featNum:
        classLabel = classLabel[0:-1]
    # 删除dataSet对应的特征
    for key in dataSet.keys():
        if featureIndex != 0:
            dataSet[key] = hstack((dataSet[key][:, 0:featureIndex], dataSet[key][:, featureIndex + 1:]))
        elif featureIndex == 0:
            dataSet[key] = dataSet[key][:, 1:]

    return dataSet, classLabel


def infoGain(dataBeforeSplit, datasAfterSplitByFeature):
    oldEnt = calEntropy(dataBeforeSplit)
    newEnt = 0.0
    for i in datasAfterSplitByFeature.keys():
        prob = float(datasAfterSplitByFeature[i].shape[0]) / dataBeforeSplit.shape[0]
        newEnt += prob * calEntropy(datasAfterSplitByFeature[i][:, -1])
    return oldEnt - newEnt


def findBestSplit(dataSet, classSet):
    dataNum, featureNum = dataSet.shape
    bestSplitFeature = -1
    mostInfoGain = 0.0
    for i in range(featureNum):
        if infoGain(classSet, datasAfterSplitByFeature(dataSet, classSet, i)) > mostInfoGain:
            bestSplitFeature = i
            mostInfoGain = infoGain(classSet, datasAfterSplitByFeature(dataSet, classSet, i))
    return bestSplitFeature


def majorClass(classSet):
    classCount = {}
    for c in classSet.tolist():
        if c[0] not in classCount.keys():
            classCount[c[0]] = 0
        classCount[c[0]] += 1
    biggestNum = 0
    for key in classCount:
        if classCount[key] > biggestNum:
            biggestNum = classCount[key]
            majorC = key
    return majorC


def ID3(dataSet, classSet, labelSet):
    if calEntropy(classSet) == 0:
        return classSet[0][0, 0]
    if dataSet.shape[1] == 0:
        return majorClass(classSet)
    bestFeature = findBestSplit(dataSet, classSet)
    bestFeatureLabel = labelSet[bestFeature]
    tree = {bestFeatureLabel: {}}
    dataSet, labelSet = deleteFeature(dataSet, classSet, labelSet, bestFeature)
    for key in dataSet.keys():
        tree[bestFeatureLabel][key] = ID3(dataSet[key][:, :-1], dataSet[key][:, -1], labelSet)
    return tree
