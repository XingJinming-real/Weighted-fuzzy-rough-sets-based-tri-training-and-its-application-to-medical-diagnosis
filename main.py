import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import preprocess
import robustModels


def main():
    """
    Implementation of Weighted fuzzy rough sets-based tri-training and its application to medical diagnosis

    fuzzyRoughSets.py: compute concepts like positive region/lower approximation related to fuzzy rough sets
    preprocess.py: normalization/preprocess data and return results
    robustModels.py: add label noise and do 10-fold cv
    sampleWeighedGradientDescend.py: update the sample weight through gradient descend
    selfClf.py: the proposed fuzzy rough sets-based tri-training model
    triTraining.py: tri-training framework used in selfClf
    :return: print accuracy in console
    """

    warnings.filterwarnings('ignore')
    dataBase = './temp/'
    normalizationMode = 'min-max'
    dataNameList = os.listdir(dataBase)
    # writer = pd.ExcelWriter('./results/results.xlsx')
    for noiseRatio in [0]:
        print(f'noisyRatio:{noiseRatio}')
        for splitRatio in [0.1, 0.2, 0.3]:
            with open('./results/' + str(noiseRatio) + '-' + str(splitRatio) + '.txt', mode='w') as f:
                # info = {}
                print(f'\tsplitRatio:{splitRatio}')
                for perDataName in dataNameList:
                    try:
                        ori, pca, dis, label = preprocess.preprocess(dataBase, perDataName, normalizationMode)
                        # get three modalities data
                        Score = []
                        splitRatioNum = 1

                        for splitRatioLoop in range(splitRatioNum):
                            shuffleState = 801
                            ori = shuffle(ori, random_state=shuffleState)
                            pca = shuffle(pca, random_state=shuffleState)
                            dis = shuffle(dis, random_state=shuffleState)
                            label = shuffle(label, random_state=shuffleState)
                            tempScore = robustModels.frsBasedClf((ori, pca, dis), label.copy(), noisyRatio=noiseRatio,
                                                                 splitRatio=splitRatio)
                            # tempScore is the accuracy of proposed model
                            Score.append(tempScore)

                        Score = np.mean(Score)
                        print(
                            f"\t{perDataName.replace('.txt', '')}\t{Score}")
                        f.write(perDataName.replace('.txt', '') + '\t' + str(Score) + '\n')
                        # info[perDataName.replace('.txt', '')] = Score
                        # infoPd = pd.Series(info)
                        # infoPd.to_excel(writer, sheet_name=str(noiseRatio) + '-' + str(splitRatio))
                        # writer.save()
                    except Exception as e:
                        f.close()
                        print(perDataName + 'Error: ', e)
                        continue
    # writer.close()


if __name__ == "__main__":
    main()
