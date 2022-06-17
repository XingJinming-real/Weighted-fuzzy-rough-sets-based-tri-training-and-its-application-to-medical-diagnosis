import time

import triTraining
from fuzzyRoughSets import getPositiveRegion
import sampleWeightGradientDescend

import numpy as np
from sklearn.model_selection import StratifiedKFold


def addLabelNoisy(dataOri, labelOri, ratio):
    """

    :param dataOri:
    :param labelOri:
    :param ratio: the ratio of noise in labelOri
    :return: new label with noise by randomly change samples' label of which are also chose randomly
    """
    if ratio == 0:
        return dataOri, labelOri

    data = dataOri.copy()
    label = labelOri.copy()
    sampleNum, _ = data.shape
    uniqueLabel = list(np.unique(label))

    idxes = np.random.choice(range(sampleNum), int(sampleNum * ratio), replace=False)
    for idx in idxes:
        uniqueLabelCopy = uniqueLabel.copy()
        uniqueLabelCopy.remove(label[idx])
        label[idx] = np.random.choice(uniqueLabelCopy)
    return data, label


def frsBasedClf(data, label, noisyRatio, splitRatio):
    """

    :param noisyRatio: the ration of noise in data
    :param splitRatio: the ratio of labeled data in trainData
    :param data: （ori,pca,dis）
    :param label:
    :return: accu
    """
    dataOri, dataPca, dataDis = data
    skf = StratifiedKFold(10, shuffle=True, random_state=801)

    testSum = 0
    correctNum = 0

    for loop, (trainIdx, testIdx) in enumerate(skf.split(dataOri, label)):
        testSum += len(testIdx)

        newLabel, trainIdx = triTraining.triTraining(dataOri, dataPca, dataDis, label.copy(), trainIdx,
                                                     splitRatio=splitRatio, noisyRatio=noisyRatio)
        # newLabel: the concatenation of predicted train label(L+U) and test label
        # trainIdx: the idx of trainData in allData

        trainDataOri = dataOri[trainIdx]
        trainNoisyLabel = newLabel[trainIdx]
        trainNoisyDataOri = trainDataOri.copy()
        uniqueLabel = np.unique(label)

        weightOri = sampleWeightGradientDescend.gradientDescend(trainNoisyDataOri, trainNoisyLabel)
        # the sample weights
        predictLabel = np.array([uniqueLabel[np.argmax(
            getPositiveRegion(dataOri[per], trainNoisyDataOri, trainNoisyLabel, weightDict=weightOri, k=3,
                              uniqueLabel=uniqueLabel)[1])] for per in testIdx])
        correctNum += np.sum(predictLabel == label[testIdx])

    return np.array(correctNum) / testSum
