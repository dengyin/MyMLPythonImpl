from numpy import *


def loadDataSetClf(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
        x = mat(dataMat)
        y = mat(labelMat).T
        y[y == 0] = -1
    return x, y


def loadDataSetReg(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
        x = mat(dataMat)
        y = mat(labelMat).T
    return x, y
