import numpy as np


def getSimilarity(x, y, kernelParameter=0.15):
    """

    :param x:
    :param y:
    :param kernelParameter:
    :return: similarity between x and y
    """
    x, y = np.array(x), np.array(y)
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * kernelParameter))


def getLowerApproximation(sampleData, perLabel, data, label, k=3, weightDict=None):
    """

    :param sampleData: pending computed sample
    :param perLabel: pending computed class
    :param data:
    :param label:
    :param k: num of neighbours of sampleData to be computed for robustness
    :param weightDict:
    :return: the lower approximation of sampleData to perLabel class
    """
    sampleNum = data.shape[0]
    if weightDict is None:
        weightDict = {i: 1 for i in range(len(data))}

    dissimilarity = {i: 1 - getSimilarity(sampleData, data[i]) for i in range(sampleNum) if
                     label[i] != perLabel and np.sum(data[i] != sampleData) != 0}
    if len(dissimilarity) == 0:
        return -1

    dissimilarity = dict(sorted(dissimilarity.items(), key=lambda x: x[1]))
    candidateKeys = list(dissimilarity.keys())[:k]
    tempMin = np.sum([weightDict[i] * dissimilarity[i] for i in candidateKeys]) / np.sum(
        [weightDict[i] for i in candidateKeys])
    return tempMin


def getPositiveRegion(sampleData, data, label, weightDict=None, k=1, uniqueLabel=None):
    """

    :param sampleData:
    :param data:
    :param label:
    :param weightDict: assigned sample weight, default None
    :param k: same with above
    :param uniqueLabel:
    :return: positive region and la to all classes
    """
    if uniqueLabel is not None:
        labelUnique = uniqueLabel
    else:
        labelUnique = np.unique(label)

    tempMax = []
    for perLabel in labelUnique:
        tempMax.append(
            getLowerApproximation(sampleData, perLabel, data, label, weightDict=weightDict, k=k))
    return max(tempMax), np.array(tempMax)
