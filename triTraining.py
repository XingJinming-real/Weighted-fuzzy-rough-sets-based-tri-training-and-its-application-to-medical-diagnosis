import robustModels
import selfClf

import numpy as np

uniqueLabel = []


def subSample(L, num):
    """
    subtract num samples from L
    :param L:
    :param num:
    :return:
    """
    for i in range(int(num)):
        L.remove(np.random.choice(L))
    return L


def measureError(clf1, p1, clf2, p2, label):
    """

    :param clf1:
    :param clf2:
    :param p1: prediction of clf1
    :param p2: prediction of clf2
    :param label: real label
    :return: the collaborated error rate
    """
    pre1 = clf1.predict(p1, uniqueLabel)
    pre2 = clf2.predict(p2, uniqueLabel)
    pre = pre1[pre1 == pre2]
    tempLabel = label[pre1 == pre2]
    if not np.sum(pre1 == pre2):
        return 0.5
    return np.sum(pre != tempLabel) / np.sum(pre1 == pre2)


def LabelU(clf, data, U):
    """
    label U
    :param clf: classifier set
    :param data:
    :param U: pending labeled data
    :return:
    """
    predict = [clf[i].predict(data[i][U[i]], uniqueLabel=uniqueLabel) for i in range(3)]
    label = [np.argmax(np.bincount(i)) for i in zip(predict[0], predict[1], predict[2])]
    # voting
    return label


def triTraining(data1, data2, data3, labelOri, trainIdx, splitRatio=0.1, noisyRatio=0.1):
    """

    :param noisyRatio: same with before
    :param splitRatio: same with before
    :param trainIdx: same with before
    :param data1: ori
    :param data2: pca
    :param data3: dis
    :param labelOri:
    :return: return learned sample weights
    """
    global uniqueLabel
    data = [data1, data2, data3]
    label = labelOri.copy()
    uniqueLabel = np.unique(label)

    clf = [selfClf.lowApproximationClf(), selfClf.lowApproximationClf(),
           selfClf.lowApproximationClf()]

    clfClfInfo = [[0.5], [0.5], [0.5]]
    # initial error rate
    trainNum = len(trainIdx)

    L = []
    U = []
    L_ = [[], [], []]
    e__ = [0.5, 0.5, 0.5]
    # new error rate
    l__ = [0, 0, 0]
    e_ = [0, 0, 0]
    # original error rate
    update = [False, False, False]

    border = int(np.ceil(trainNum * splitRatio))
    while True:
        if any(label[trainIdx[:border]] - label[trainIdx[border - 1]]):
            break
        np.random.shuffle(trainIdx)
    _, noisyLabel = robustModels.addLabelNoisy(data1[trainIdx[:border]], label[trainIdx[:border]], noisyRatio)
    label[trainIdx[:border]] = noisyLabel
    for i in range(3):
        L.append(trainIdx[:border].tolist())
        U.append(trainIdx[border:])
        clf[i].fit(data[i][L[i]], label[L[i]])
        # initialize base classifiers

    """tri-training"""
    while True:
        for i in range(3):
            L_[i] = []
            update[i] = False
            e_[i] = measureError(clf[(i + 1) % 3], data[(i + 1) % 3][trainIdx[:border]], clf[(i + 2) % 3],
                                 data[(i + 2) % 3][trainIdx[:border]], label[trainIdx[:border]].copy())
            clfClfInfo[i].append(e_[i])
            if e_[i] < e__[i]:
                for possibleIdx in U[0]:
                    if clf[(i + 1) % 3].predict(data[(i + 1) % 3][possibleIdx].reshape(1, -1), uniqueLabel) \
                            == clf[(i + 2) % 3].predict(data[(i + 2) % 3][possibleIdx].reshape(1, -1), uniqueLabel, 3):
                        L_[i].append(possibleIdx)
                if not l__[i]:
                    l__[i] = np.floor(e_[i] / (e__[i] - e_[i]) + 1)
                if l__[i] < len(L_[i]):
                    if e_[i] * len(L_[i]) < e__[i] * l__[i]:
                        update[i] = True
                    elif l__[i] > e_[i] / (e__[i] - e_[i]):
                        L_[i] = subSample(L_[i], np.ceil(e__[i] * l__[i] / e_[i] - 1))
                        update[i] = True
        for i in range(3):
            if update[i]:
                tempL = L[i].copy()
                tempL.extend(L_[i])
                tempLabel = label[tempL]
                for perI in L_[i]:
                    tempLabel[tempL.index(perI)] = clf[(i + 1) % 3].predict(data[(i + 1) % 3][perI].reshape(1, -1),
                                                                            uniqueLabel)[0]
                clf[i].fit(data[i][tempL], tempLabel)
                e__[i] = e_[i]
                l__[i] = len(L_[i])

        if not any(update):
            break

    labelU = LabelU(clf, data, U)
    # predict U label
    label[trainIdx[border:]] = labelU
    return label, trainIdx
