import numpy
from numpy import *


def loadDataSet(fileName):  # general function to parse tab -delimited floats
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
    m = mat(dataMat).shape[0]
    n = mat(dataMat).shape[1]
    return numpy.c_[ones((m, 1)), mat(dataMat)], mat(labelMat).T
