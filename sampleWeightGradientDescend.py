import numpy as np


def getDistance(x, y):
    """

    :param x:
    :param y:
    :return: Euclidean distance
    """
    x, y = np.array(x), np.array(y)
    return np.linalg.norm(x - y)


def getNearestKSample(sampleIdx, data, k):
    """

    :param sampleIdx: anchor sample
    :param data:
    :param k: num of needed sample
    :return:
    """
    distance = [getDistance(data[sampleIdx], data[i]) for i in range(len(data))]
    argDistance = np.argsort(distance)
    return argDistance[1:k + 1]


def getKHandM(sampleIdx, data, label, k=1, badPoints=None):
    """
    get nearest k hit/miss samples
    :param sampleIdx: anchor sample
    :param data:
    :param label:
    :param k:
    :param badPoints:
    :return: sample idxes
    """
    if badPoints is None:
        badPoints = []
    distance = [getDistance(data[sampleIdx], data[i]) for i in range(len(data))]
    argDistance = np.argsort(distance)
    nearestHit = []
    nearestMiss = []
    for i in argDistance:
        if len(nearestHit) == k and len(nearestMiss) == k:
            return nearestHit, nearestMiss
        if i == sampleIdx:
            continue
        if label[i] == label[sampleIdx] and len(nearestHit) < k and i not in badPoints:
            nearestHit.append(i)
            continue
        if label[i] != label[sampleIdx] and len(nearestMiss) < k and i not in badPoints:
            nearestMiss.append(i)
            continue
    return nearestHit, nearestMiss


def getBadPoints(data, label, k):
    """
    literally get bad-points
    :param data:
    :param label:
    :param k: the num of neighbours to be considered
    :return: bad-points idxes
    """
    badPoints = []
    label = np.array(label)
    for i in range(len(data)):
        sameLabelNum = list(label[getNearestKSample(i, data, k)]).count(label[i])
        if sameLabelNum <= k * 0.25:
            badPoints.append(i)
    return badPoints


def Loss(data, weight, nearestSamples, badPoints=None, kernelP=0.15):
    """

    :param data:
    :param weight:
    :param nearestSamples: samples' idxes
    :param badPoints:
    :param kernelP: the parameter of kernel function
    :return:
    """
    if badPoints is None:
        badPoints = []
    loss = 0
    for i in range(len(data)):
        if i not in badPoints:
            for j in range(len(nearestSamples[i][1])):
                loss += 1 - np.exp(-weight[nearestSamples[i][1][j]] ** 2 * getDistance(data[i], data[
                    nearestSamples[i][1][j]]) ** 2 / kernelP)
    return loss


def gradientDescend(data, label, k=5, weight=None):
    """

    :param data:
    :param label:
    :param k: the number of neighbours
    :param weight: the data weight
    :return: sample weights
    """
    kernelParameter = 0.15
    sampleNum, featureNum = data.shape

    if weight is not None:
        weight = weight
    else:
        weight = {i: 1 for i in range(sampleNum)}
        # initialize weight if not given

    badPoints = getBadPoints(data, label, k=9)
    # get badPoints
    nearestSample = {i: getKHandM(i, data, label, k=k, badPoints=badPoints) for i in range(sampleNum) if
                     i not in badPoints}

    lr = 0.1
    threshold = 0.001
    loss = Loss(data, weight, nearestSample, badPoints=badPoints)
    iterNum = 0
    while True:
        for i in range(sampleNum):
            if i in badPoints:
                continue
            for j in range(len(nearestSample[i][1])):
                weight[nearestSample[i][1][j]] += lr * 2 / kernelParameter * getDistance(data[i], data[
                    nearestSample[i][1][j]]) ** 2 * weight[nearestSample[i][1][j]] * np.exp(
                    -(weight[nearestSample[i][1][j]] * getDistance(data[i], data[
                        nearestSample[i][1][j]])) ** 2 / kernelParameter)

        temp = Loss(data, weight, nearestSample, badPoints=badPoints)
        iterNum += 1
        if abs(temp - loss) < threshold or iterNum > 100:
            break
        else:
            loss = temp

    for i in badPoints:
        weight[i] = 1e-6
    return weight
