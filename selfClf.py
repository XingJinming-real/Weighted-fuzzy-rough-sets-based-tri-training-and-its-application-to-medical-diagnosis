import sampleWeightGradientDescend
from fuzzyRoughSets import *


class lowApproximationClf:
    def __init__(self):
        self.data = None
        self.label = None
        self.weight = None
        # the num of L
        self.lLen = None

    def fit(self, data, label):
        self.data = data
        self.label = label
        weight = {i: 1 for i in range(len(data))}
        if self.lLen:
            # it means it's not the first time we train the model,
            # and we should take the last weights as this round initialization
            for i in range(self.lLen):
                weight[i] = self.weight[i]
            weight, weightNorm, badPoints, badPointsErrorLabel = sampleWeightGradientDescend.gradientDescend(
                self.data, self.label, weight=weight)
            # update sample weights
        self.weight = weight
        if self.lLen is None:
            self.lLen = len(self.weight)

    def predict(self, data, uniqueLabel, weight=None):
        if weight is not None:
            self.weight = weight
        data = np.array(data)
        if len(data) == 1:
            data = data.reshape((1, -1))
        predictLabel = []
        for i, perData in enumerate(data):
            predictLabel.append(
                uniqueLabel[np.argmax(getPositiveRegion(perData, self.data, self.label, weightDict=self.weight, k=3,
                                                        uniqueLabel=uniqueLabel)[1])])
        return predictLabel
