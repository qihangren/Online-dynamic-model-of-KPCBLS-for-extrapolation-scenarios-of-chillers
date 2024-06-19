# -*- coding: utf-8 -*-
"""
Demo_BLS

Author:
Date: 2024-05-20
"""

import pandas as pd
import numpy as np
import scipy.io as scio
from BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes

# region Dataset processing
dataset7_28 = pd.read_csv('../7-28.csv')
dataset7_30 = pd.read_csv('../7-30.csv')
dataset7_32 = pd.read_csv('../7-32.csv')
dataset7_28 = np.array(dataset7_28)
dataset7_30 = np.array(dataset7_30)
dataset7_32 = np.array(dataset7_32)
dataset2832 = np.vstack((dataset7_28, dataset7_32))  # Extrapolation scenario
dataset283230fp = np.vstack((dataset7_28, dataset7_32, dataset7_30[0:36]))  # Online scenarios. 36 sampling points need to be added sequentially, divided into 10 batches.
traindata = dataset2832[:, 0:5]  # Input parameters in the training dataset for extrapolation scenarios. X = [ Tew,L, Tew,E, Tcw,E, Gew, Gcw ]
trainlabel = dataset2832[:, 5:7]  # Output parameters in the training dataset for extrapolation scenarios. Y = [ Tcw,L, P ]
COP_train_true = dataset2832[:, 7]  # COP calculated through observation values in extrapolation scenarios. As an auxiliary variable
#traindata = dataset283230fp[:, 0:5]  # Input parameters in the training dataset for online scenarios. X = [ Tew,L, Tew,E, Tcw,E, Gew, Gcw ]
#trainlabel = dataset283230fp[:, 5:7]  # Output parameters in the training dataset for online scenarios. Y = [ Tcw,L, P ]
#COP_train_true = dataset283230fp[:, 7]  # COP calculated through observation values in extrapolation scenarios. As an auxiliary variable
testdata = dataset7_30[:, 0:5]  # Input parameters in the testing dataset for online scenarios. X = [ Tew,L, Tew,E, Tcw,E, Gew, Gcw ]
testlabel = dataset7_30[:, 5:7]  # Output parameters in the testing dataset for extrapolation scenarios. Y = [ Tcw,L, P ]
COP_test_true = dataset7_30[:, 7]  # COP calculated through observation values in extrapolation scenarios. As an auxiliary variable
# endregion

# region Initialize BLS parameters
N1 = 10  # Number of nodes in each feature window
N2 = 10  # Number of windows in the feature layer
N3 = 300  # Number of nodes in the enhancement layer
s = 0.5  # shrink coefficient
c = 2**-30  # Regularization coefficient
# endregion

print('-------------------BLS---------------------------')
BLS(traindata, trainlabel, testdata, testlabel, s, c, N1, N2, N3, dataset2832, dataset7_30, COP_train_true, COP_test_true)
# dataset2832 is used in Scenario 1, and dataset283230fp is used in Scenario 2


