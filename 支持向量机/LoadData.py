from numpy import *


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = [];
    classMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(curLine[i])
        dataMat.append(lineArr)
        classMat.append(curLine[-1])
    m = mat(dataMat).shape[0]
    n = mat(dataMat).shape[1]
    return mat(dataMat).astype('float64'), mat(classMat).T.astype('float64')
