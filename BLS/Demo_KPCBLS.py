# -*- coding: utf-8 -*-
"""
Demo_KPCBLS

Author: 
Date: 2024-05-20
"""

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# region Activation function, pseudo inverse function, loss function
def tansig(x):
    return (2/(1+np.exp(-2*x)))-1
def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))
def sigmoid_derivative(data):
    sig = sigmoid(data)
    return sig * (1 - sig)
def linear(data):
    return data
def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
def pinv(A, reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z
def sparse_bls(A, b):
    lam = 0.001  # Sparsity parameter, controlling the degree of sparsity
    itrs = 50    # Iterations
    AA = A.T.dot(A)  # calculate A^T * A
    m = A.shape[1]   # Obtain the number of columns for A and b
    n = b.shape[1]
    x1 = np.zeros([m, n])  # initialize variable
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I  # Calculate the pseudo inverse of a matrix
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):  # Iterative update of weight matrix
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk
def custom_loss(x_true, y_true, y_pred):  # Custom loss function, The calculated result is a number
    mse_TCO = mean_squared_error(y_true[:, 0], y_pred[:, 0])  # calculate  MSE_TCO
    mse_P = mean_squared_error(y_true[:, 1], y_pred[:, 1])    # calculate  MSE_P
    plr = 4.187 * x_true[:,3] * (x_true[:,1] - x_true[:,0]) / 6330
    loss_en = (6330 * plr.reshape(-1, 1) +
               y_pred[:, 1].reshape(-1, 1) -
               4.187 * x_true[:, 4] *
               (y_pred[:, 0].reshape(-1, 1) - x_true[:, 2].reshape(-1, 1)))
    loss_en = np.where((loss_en >= -760) & (loss_en <= 760), 0.0,
                           np.where(loss_en < -760, -760 - loss_en, loss_en - 760))
    loss_energy = np.mean(loss_en) / 2  # Calculate average energy loss (Also known as physical inconsistency loss)
    loss = mse_TCO + mse_P + loss_energy
    return loss
# endregion

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

# region Initialize KPCBLS parameters
N1 = 10  # Number of nodes in each feature window
N2 = 10  # Number of windows in the feature layer
N3 = 300  # Number of nodes in the enhancement layer
s = 0.5  # shrink coefficient
c = 2**-30  # Regularization coefficient
train_losses = []   # Store training losses
biasOfInputData = 0.1 * np.ones((traindata.shape[0], 1))  # Initialize the bias of the input layer
WeightOfFeatureLayer = np.ones((traindata.shape[1] + 1, N1 * N2))  # Initialize the weights of feature layers
biasOfFeatureLayer = 0.1 * np.ones((traindata.shape[0], 1))  # Initialize the bias of feature layers
weightOfEnhanceLayer = 0.1 * np.ones((N1 * N2 + 1, N3))  # Initialize the weights of the enhancement layer
distOfMaxAndMin = np.ones((N2,N1))  # The difference between the maximum and minimum values of elements in each window of the feature layer
minOfEachWindow = np.ones((N2,N1))  # The minimum value of elements in each window of the feature layer
# endregion

# region Training model
# region The hyperparameter initialization of the Adam optimizer. First and second moment initialization
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.01
m_dw_input_feat = np.zeros_like(WeightOfFeatureLayer)  # The initialization of the first moment of the weight (input layer to feature layer)
v_dw_input_feat = np.zeros_like(WeightOfFeatureLayer)  # The initialization of the second moment of the weight (input layer to feature layer)
m_db_input_feat = np.zeros_like(biasOfInputData)  # The initialization of the first moment of the paranoia (input layer to feature layer)
v_db_input_feat = np.zeros_like(biasOfInputData)  # The initialization of the second moment of the paranoia (input layer to feature layer)
m_dw_feat_enhan = np.zeros_like(weightOfEnhanceLayer)  # The initialization of the first moment of the weight (feature layer to enhancement layer)
v_dw_feat_enhan = np.zeros_like(weightOfEnhanceLayer)  # The initialization of the sceond moment of the weight (feature layer to enhancement layer)
m_db_feat_enhan = np.zeros_like(biasOfFeatureLayer)  # The initialization of the first moment of the paranoia (feature layer to enhancement layer)
v_db_feat_enhan = np.zeros_like(biasOfFeatureLayer)  # The initialization of the second moment of the paranoia (feature layer to enhancement layer)
# endregion

# region First Forward Propagation, Error Backpropagation, Adam Optimization
# region Forward Propagation
train_x = preprocessing.scale(traindata, axis=1)  # Standardize each row to have a mean of 0 and a standard deviation of 1, in order to eliminate dimensional differences between different eigenvalues
FeatureOfInputDataWithBias = np.hstack([train_x, biasOfInputData])  # Choosing 0.1 here may be to ensure that the bias term does not have a significant impact on the model when the data is small, while also providing a certain degree of bias correction.
OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])
time_start = time.time()  # Timing begins
for i in range(N2):
    random.seed(i)  # Generate different random numbers for each window
    weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1,N1) - 1 # The purpose is to initialize the values of the weight matrix to a smaller range for faster training. Generally speaking, smaller initial weights can avoid gradient vanishing or exploding problems, and help the network converge to the appropriate solution faster.
    FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
    scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
    FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
    betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
    start_col = i * N1
    end_col = (i + 1) * N1
    WeightOfFeatureLayer[:, start_col:end_col] = betaOfEachWindow
    outputOfEachWindow = FeatureOfInputDataWithBias.dot(betaOfEachWindow)
    distOfMaxAndMin[i,:] = np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0)
    minOfEachWindow[i,:] = np.min(outputOfEachWindow, axis=0)
    outputOfEachWindow = (outputOfEachWindow -minOfEachWindow[i,:]) / distOfMaxAndMin[i,:]
    OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
    del outputOfEachWindow
    del FeatureOfEachWindow
    del weightOfEachWindow

InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, biasOfFeatureLayer])

if N1 * N2 >= N3:
    random.seed(67797325)
    weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
else:
    random.seed(67797325)
    weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1,N3).T - 1).T
    # The purpose of orthogonalization is to ensure that the weight matrix has better numerical properties, such as reducing redundant information, avoiding gradient vanishing, etc., which helps with stable training and better performance.

tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
OutputOfEnhanceLayer = sigmoid(tempOfOutputOfEnhanceLayer * parameterOfShrink)
InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
pinvOfInput = pinv(InputOfOutputLayer, c)
OutputWeight = np.dot(pinvOfInput, trainlabel)
OutputOfTrain = np.array(np.dot(InputOfOutputLayer, OutputWeight))
# endregion
# region Error Backpropagation
# dataset2832 is used in Scenario 1, and dataset283230fp is used in Scenario 2
loss_0 = (6330 * dataset2832[:, 8].reshape(-1, 1) +
          OutputOfTrain[:, 1].reshape(-1, 1) -
          4.187 * traindata[:, 4].reshape(-1, 1) *
          (OutputOfTrain[:, 0].reshape(-1, 1) - traindata[:, 2].reshape(-1, 1)))
loss_0 = np.where((loss_0 >= -760) & (loss_0 <= 760), 0.0,
                  np.where(loss_0 < -760, -760 - loss_0, loss_0 - 760))
output_error = OutputOfTrain - trainlabel + loss_0
FeatEnhan_error = np.dot(output_error, OutputWeight.T)
enhan_error = FeatEnhan_error[:, N1*N2:]
feat_error = FeatEnhan_error[:,:N1*N2]
# endregion
# region Adam optimizer
m_dw_feat_enhan = beta1 * m_dw_feat_enhan + (1 - beta1) * (np.dot(InputOfEnhanceLayerWithBias.T, enhan_error))
v_dw_feat_enhan = beta2 * v_dw_feat_enhan + (1 - beta2) * (np.square(np.dot(InputOfEnhanceLayerWithBias.T, enhan_error)))
m_db_feat_enhan = beta1 * m_db_feat_enhan + (1 - beta1) * (np.sum(enhan_error, axis=1).reshape(-1,1))
v_db_feat_enhan = beta2 * v_db_feat_enhan + (1 - beta2) * (np.square(np.sum(enhan_error, axis=1).reshape(-1,1)))

m_dw_input_feat = beta1 * m_dw_input_feat + (1 - beta1) * (np.dot(FeatureOfInputDataWithBias.T, feat_error))
v_dw_input_feat = beta2 * v_dw_input_feat + (1 - beta2) * (np.square(np.dot(FeatureOfInputDataWithBias.T, feat_error)))
m_db_input_feat = beta1 * m_db_input_feat + (1 - beta1) * (np.sum(feat_error, axis=1).reshape(-1,1))
v_db_input_feat = beta2 * v_db_input_feat + (1 - beta2) * (np.square(np.sum(feat_error, axis=1).reshape(-1,1)))

m_dw_feat_enhan_corrected = m_dw_feat_enhan / (1 - beta1)
v_dw_feat_enhan_corrected = v_dw_feat_enhan / (1 - beta2)
m_db_feat_enhan_corrected = m_db_feat_enhan / (1 - beta1)
v_db_feat_enhan_corrected = v_db_feat_enhan / (1 - beta2)

m_dw_input_feat_corrected = m_dw_input_feat / (1 - beta1)
v_dw_input_feat_corrected = v_dw_input_feat / (1 - beta2)
m_db_input_feat_corrected = m_db_input_feat / (1 - beta1)
v_db_input_feat_corrected = v_db_input_feat / (1 - beta2)

weightOfEnhanceLayer -= learning_rate * m_dw_feat_enhan_corrected / (
        np.sqrt(v_dw_feat_enhan_corrected) + epsilon)
biasOfFeatureLayer -= learning_rate * m_db_feat_enhan_corrected / (
        np.sqrt(v_db_feat_enhan_corrected) + epsilon)
WeightOfFeatureLayer -= learning_rate * m_dw_input_feat_corrected / (
        np.sqrt(v_dw_input_feat_corrected) + epsilon)
biasOfInputData -= learning_rate * m_db_input_feat_corrected / (
        np.sqrt(v_db_input_feat_corrected) + epsilon)
# endregion
# endregion

# region Define stop conditions
desired_loss = 90  # Stop iteration when loss is less than or equal to desired_loss
max_iterations = 700  # Stop iteration when max_iterations iterations are reached
# endregion

# region Define variables and store the optimal value in iterations
min_loss = float('inf')
best_weightOfInput = None
best_weightOfEnhanceLayer = None
best_bias_input_feat = None
best_bias_feat_enhan= None
best_parameterOfShrink = None
best_distOfMaxAndMin = None
best_minOfEachWindow = None
best_OutputWeight = None
best_y_train = None
# endregion

# region N-th forward propagation, error backpropagation, Adam optimization
for j in range(max_iterations):
    # region forward propagation
    for k in range(N2):
        outputOfEachWindow = FeatureOfInputDataWithBias.dot(WeightOfFeatureLayer[:, N1 * k:N1 * (k + 1)])
        distOfMaxAndMin[k, :] = np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0)
        minOfEachWindow[k, :] = np.min(outputOfEachWindow, axis=0)
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[k, :]) / distOfMaxAndMin[k, :]
        OutputOfFeatureMappingLayer[:, N1*k:N1*(k+1)] = outputOfEachWindow
        del outputOfEachWindow
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, biasOfFeatureLayer])
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = sigmoid(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput, trainlabel)
    OutputOfTrain = np.array(np.dot(InputOfOutputLayer, OutputWeight))
    # endregion
    # region  Calculate Customloss
    loss = custom_loss(traindata, trainlabel, OutputOfTrain)
    print("Iteration {}, Loss: {}".format(j + 1, loss))
    train_losses.append(loss)
    # endregion
    # region If the current loss value is smaller than the minimum loss, update the minimum loss and corresponding weights and biases
    if loss < min_loss:
        min_loss = loss
        best_weightOfInput = WeightOfFeatureLayer.copy()
        best_weightOfEnhanceLayer = weightOfEnhanceLayer.copy()
        best_bias_input_feat = biasOfInputData.copy()
        best_bias_feat_enhan = biasOfFeatureLayer.copy()
        best_distOfMaxAndMin = distOfMaxAndMin.copy()
        best_minOfEachWindow = minOfEachWindow.copy()
        best_parameterOfShrink = parameterOfShrink.copy()
        best_OutputWeight = OutputWeight.copy()
        best_y_train = OutputOfTrain.copy()
    # endregion
    # region Check if the stop condition has been met
    if loss <= desired_loss:
        print("Desired loss reached. Stopping training.")
        break  # Directly jump out of the for loop without performing error backpropagation and Adam optimization below
    # endregion
    # region error backpropagation
    # dataset2832 is used in Scenario 1, and dataset283230fp is used in Scenario 2
    loss_0 = (6330 * dataset2832[:, 8].reshape(-1, 1) +
              OutputOfTrain[:, 1].reshape(-1, 1) -
              4.187 * traindata[:, 4].reshape(-1, 1) *
              (OutputOfTrain[:, 0].reshape(-1, 1) - traindata[:, 2].reshape(-1, 1)))
    loss_0 = np.where((loss_0 >= -760) & (loss_0 <= 760), 0.0,
                      np.where(loss_0 < -760, -760 - loss_0, loss_0 - 760))
    output_error = OutputOfTrain - trainlabel + loss_0
    FeatEnhan_error = np.dot(output_error, OutputWeight.T)
    enhan_error = FeatEnhan_error[:, N1 * N2:]
    feat_error = FeatEnhan_error[:, :N1 * N2]
    # endregion
    # region Adam optimization
    m_dw_feat_enhan = beta1 * m_dw_feat_enhan + (1 - beta1) * (np.dot(InputOfEnhanceLayerWithBias.T, enhan_error))
    v_dw_feat_enhan = beta2 * v_dw_feat_enhan + (1 - beta2) * (
        np.square(np.dot(InputOfEnhanceLayerWithBias.T, enhan_error)))
    m_db_feat_enhan = beta1 * m_db_feat_enhan + (1 - beta1) * (np.sum(enhan_error, axis=1).reshape(-1, 1))
    v_db_feat_enhan = beta2 * v_db_feat_enhan + (1 - beta2) * (np.square(np.sum(enhan_error, axis=1).reshape(-1, 1)))

    m_dw_input_feat = beta1 * m_dw_input_feat + (1 - beta1) * (np.dot(FeatureOfInputDataWithBias.T, feat_error))
    v_dw_input_feat = beta2 * v_dw_input_feat + (1 - beta2) * (
        np.square(np.dot(FeatureOfInputDataWithBias.T, feat_error)))
    m_db_input_feat = beta1 * m_db_input_feat + (1 - beta1) * (np.sum(feat_error, axis=1).reshape(-1, 1))
    v_db_input_feat = beta2 * v_db_input_feat + (1 - beta2) * (np.square(np.sum(feat_error, axis=1).reshape(-1, 1)))

    m_dw_feat_enhan_corrected = m_dw_feat_enhan / (1 - beta1)
    v_dw_feat_enhan_corrected = v_dw_feat_enhan / (1 - beta2)
    m_db_feat_enhan_corrected = m_db_feat_enhan / (1 - beta1)
    v_db_feat_enhan_corrected = v_db_feat_enhan / (1 - beta2)

    m_dw_input_feat_corrected = m_dw_input_feat / (1 - beta1)
    v_dw_input_feat_corrected = v_dw_input_feat / (1 - beta2)
    m_db_input_feat_corrected = m_db_input_feat / (1 - beta1)
    v_db_input_feat_corrected = v_db_input_feat / (1 - beta2)

    weightOfEnhanceLayer -= learning_rate * m_dw_feat_enhan_corrected / (
            np.sqrt(v_dw_feat_enhan_corrected) + epsilon)
    biasOfFeatureLayer -= learning_rate * m_db_feat_enhan_corrected / (
            np.sqrt(v_db_feat_enhan_corrected) + epsilon)
    WeightOfFeatureLayer -= learning_rate * m_dw_input_feat_corrected / (
            np.sqrt(v_dw_input_feat_corrected) + epsilon)
    biasOfInputData -= learning_rate * m_db_input_feat_corrected / (
            np.sqrt(v_db_input_feat_corrected) + epsilon)
    # endregion
time_end = time.time()
Training_time = time_end - time_start
print('PCBLS Training has been finished!')
print('The Total PCBLS Training Time is : ', round(Training_time, 6), ' seconds')
# endregion
# endregion

# region Evaluation indicators for training results &  Store results
# The evaluation indicators and training errors corresponding to the optimal iteration result
mae_BLS_Train_TCO = mean_absolute_error(trainlabel[:, 0], best_y_train[:, 0])  # calculate MAE_TCO
mae_BLS_Train_P = mean_absolute_error(trainlabel[:, 1], best_y_train[:, 1])  # calculate MAE_P
rmse_BLS_Train_TCO = np.sqrt(mean_squared_error(trainlabel[:, 0], best_y_train[:, 0]))  # calculate RMSE_TCO
rmse_BLS_Train_P = np.sqrt(mean_squared_error(trainlabel[:, 1], best_y_train[:, 1]))  # calculate RMSE_P
r2_BLS_Train_TCO = r2_score(trainlabel[:, 0], best_y_train[:, 0])  # calculate  R^2_TCO
r2_BLS_Train_P = r2_score(trainlabel[:, 1], best_y_train[:, 1])  # calculate R^2_P
mape_BLS_Train_TCO = np.mean(np.abs((trainlabel[:,0], best_y_train[:,0]) / trainlabel[:,0])) * 100  # calculate MAPE_TCO
mape_BLS_Train_P = np.mean(np.abs((trainlabel[:,1], best_y_train[:,1]) / trainlabel[:,1])) * 100  # calculate MAPE_TCO
mse_BLS_Train_TCO = mean_squared_error(trainlabel[:, 0], best_y_train[:, 0])  # calculate MSE_TCO
mse_BLS_Train_P = mean_squared_error(trainlabel[:, 1], best_y_train[:, 1])  # calculate MSE_P
# dataset2832 is used in Scenario 1, and dataset283230fp is used in Scenario 2
loss_0 = (6330 * dataset2832[:, 8].reshape(-1, 1) +
          best_y_train[:, 1].reshape(-1, 1) -
          4.187 * traindata[:, 4].reshape(-1, 1) *
          (best_y_train[:, 0].reshape(-1, 1) - traindata[:, 2].reshape(-1, 1)))
loss_0 = np.where((loss_0 >= -760) & (loss_0 <= 760), 0.0,
                  np.where(loss_0 < -760, -760 - loss_0, loss_0 - 760))
loss_energy_BLS_Train = np.mean(loss_0) / 2
loss_BLS_Train = mse_BLS_Train_TCO + mse_BLS_Train_P + loss_energy_BLS_Train
print('Training accuracy indicators')
print("MAE KPCBLS Train TCO:", mae_BLS_Train_TCO)
print("MAE KPCBLS Train P:", mae_BLS_Train_P)
print("RMSE KPCBLS Train TCO:", rmse_BLS_Train_TCO)
print("RMSE KPCBLS Train P:", rmse_BLS_Train_P)
print("R^2 KPCBLS Train TCO:", r2_BLS_Train_TCO)
print("R^2 KPCBLS Train P:", r2_BLS_Train_P)
print("MAPE KPCBLS Train TCO:", mape_BLS_Train_TCO)
print("MAPE KPCBLS Train P:", mape_BLS_Train_P)
print("MSE KPCBLS Train TCO:", mse_BLS_Train_TCO)
print("MSE KPCBLS Train P:", mse_BLS_Train_P)
print("loss_energy KPCBLS Train:", loss_energy_BLS_Train)
print("loss KPCBLS Train:", loss_BLS_Train)
COP_train = 6330 * dataset2832[:, 8] / best_y_train[:, 1]
COP_train = pd.DataFrame(COP_train)
#COP_train.to_csv('NETOUT_TRAIN_KPCBLS_COP_fp_1.csv')
best_y_train = pd.DataFrame(best_y_train)
#best_y_train.to_csv('NETOUT_TRAIN_KPCBLS_TCO_P_fp_1.csv')
print("Training results:")
print(best_y_train)
# endregion

# region Testing model
# Test using the weights and biases corresponding to the minimum loss after the loop ends
# Randomly select an index of 360 elements from best_bias_input_feat
random_indices_input_feat = np.random.choice(best_bias_input_feat.shape[0], testdata.shape[0], replace=False)
bias_input_feat_test = best_bias_input_feat[random_indices_input_feat]  # Retrieve corresponding elements based on random indexes
bias_input_feat_test = bias_input_feat_test.reshape((testdata.shape[0], 1))
random_indices_feat_enhan = np.random.choice(best_bias_feat_enhan.shape[0], testdata.shape[0], replace=False)
bias_feat_enhan_test = best_bias_feat_enhan[random_indices_feat_enhan]
bias_feat_enhan_test = bias_feat_enhan_test.reshape((testdata.shape[0], 1))
test_x = preprocessing.scale(testdata, axis=1)
a = 0.1 * np.ones((test_x.shape[0],1))
FeatureOfInputDataWithBiasTest = np.hstack([test_x, a])
OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
b = 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))
for m in range(N2):
    outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, best_weightOfInput[:, N1 * m:N1 * (m + 1)])
    OutputOfFeatureMappingLayerTest[:, N1 * m:N1 * (m + 1)] = (
                outputOfEachWindowTest - best_minOfEachWindow[m,:]) / best_distOfMaxAndMin[m,:]
InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, b])
tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, best_weightOfEnhanceLayer)
OutputOfEnhanceLayerTest = sigmoid(tempOfOutputOfEnhanceLayerTest * best_parameterOfShrink)
InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
y_test = np.array(np.dot(InputOfOutputLayerTest, best_OutputWeight))
# endregion

# region Prior knowledge model
#a = [185.37, 612.13, -391.66, 678.43]  # Prior knowledge of 6-28
#b = [-2.1865, 43.477, -78.387, 64.018, -21.098]  # Prior knowledge of 6-28
a = [278.41, 242.18, 276.2, 303.78]  # Prior knowledge of 7_30
b = [-3.2827, 46.55, -82.731, 66.104, -20.9]  #Prior knowledge of 7_30
QE = 4.187 * testdata[:,3].reshape(-1,1) * (testdata[:,1].reshape(-1,1) - testdata[:,0].reshape(-1,1))
plr = QE / 6330
power = a[0] + a[1] * plr + a[2] * plr ** 2 + a[3] * plr ** 3
cop = b[0] + b[1] * plr + b[2] * plr ** 2 + b[3] * plr ** 3 + b[4] * plr ** 4
tco = (QE + power) / (4.187 * testdata[:,4].reshape(-1,1)) + testdata[:,2].reshape(-1,1)
y_knowledge = np.hstack((tco, power))
y_test = 0.5 * y_test + 0.5 * y_knowledge
# endregion

# region Evaluation indicators for testing results & Store results
# Error between y_test and Y_OB
mae_BLS_Test_TCO = mean_absolute_error(testlabel[:, 0], y_test[:, 0])
mae_BLS_Test_P = mean_absolute_error(testlabel[:, 1], y_test[:, 1])
rmse_BLS_Test_TCO = np.sqrt(mean_squared_error(testlabel[:, 0], y_test[:, 0]))
rmse_BLS_Test_P = np.sqrt(mean_squared_error(testlabel[:, 1], y_test[:, 1]))
r2_BLS_Test_TCO = r2_score(testlabel[:, 0], y_test[:, 0])
r2_BLS_Test_P = r2_score(testlabel[:, 1], y_test[:, 1])
mape_BLS_Test_TCO = np.mean(np.abs((testlabel[:,0], y_test[:, 0]) / testlabel[:, 0])) * 100
mape_BLS_Test_P = np.mean(np.abs((testlabel[:,1], y_test[:, 1]) / testlabel[:, 1])) * 100
mse_BLS_Test_TCO = mean_squared_error(testlabel[:, 0], y_test[:, 0])
mse_BLS_Test_P = mean_squared_error(testlabel[:, 1], y_test[:, 1])
loss_0 = (6330 * dataset7_30[:, 8].reshape(-1, 1) +
          y_test[:, 1].reshape(-1, 1) -
          4.187 * testdata[:, 4].reshape(-1, 1) *
          (y_test[:, 0].reshape(-1, 1) - testdata[:, 2].reshape(-1, 1)))
loss_0 = np.where((loss_0 >= -760) & (loss_0 <= 760), 0.0,
                  np.where(loss_0 < -760, -760 - loss_0, loss_0 - 760))
loss_energy_BLS_Test = np.mean(loss_0) / 2
loss_BLS_Test = mse_BLS_Test_TCO + mse_BLS_Test_P + loss_energy_BLS_Test
error_Test = np.array([mae_BLS_Test_TCO, rmse_BLS_Test_TCO, r2_BLS_Test_TCO,
                       mae_BLS_Test_P, rmse_BLS_Test_P, r2_BLS_Test_P,
                       loss_energy_BLS_Test, Training_time]).reshape(1, -1)
error_Test = pd.DataFrame(error_Test)
error_Test.to_csv('NETOUT_TEST_KPCBLS_error_fp_1.csv')
print("Testing accuracy indicators")
print("MAE KPCBLS Test TCO:", mae_BLS_Test_TCO)
print("MAE KPCBLS Test P:", mae_BLS_Test_P)
print("RMSE KPCBLS Test TCO:", rmse_BLS_Test_TCO)
print("RMSE KPCBLS Test P:", rmse_BLS_Test_P)
print("R^2 KPCBLS Test TCO:", r2_BLS_Test_TCO)
print("R^2 KPCBLS Test P:", r2_BLS_Test_P)
print("MAPE KPCBLS Test TCO:", mape_BLS_Test_TCO)
print("MAPE KPCBLS Test P:", mape_BLS_Test_P)
print("MSE KPCBLS Test TCO:", mse_BLS_Test_TCO)
print("MSE KPCBLS Test P:", mse_BLS_Test_P)
print("loss_energy KPCBLS Test:", loss_energy_BLS_Test)
print("loss KPCBLS Test:", loss_BLS_Test)
COP_test = 6330 * dataset7_30[:, 8] / y_test[:, 1]
COP_test = pd.DataFrame(COP_test)
#COP_test.to_csv('NETOUT_TEST_KPCBLS_COP_fp_1.csv')
y_test = pd.DataFrame(y_test)
#y_test.to_csv('NETOUT_TEST_KPCBLS_TCO_P_fp_1.csv')
print("Testing results:")
print(y_test)
# endregion

# region Plotting
# Create a figure and axis objects with a layout of (2, 2)
fig, axs = plt.subplots(3, 2, figsize=(10, 12))

# Scatter plot for train data
axs[0, 0].scatter(trainlabel[:, 0], best_y_train.iloc[:, 0])
axs[0, 0].set_xlabel('TCO_train_ture')
axs[0, 0].set_ylabel('TCO_train_pred')
axs[0, 0].set_title('TCO_train_ture & TCO_train_pred')

# Calculate and annotate R^2 for train data
r2_train_tco = r2_score(trainlabel[:, 0], best_y_train.iloc[:, 0])
axs[0, 0].annotate(f'R^2 = {r2_train_tco:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

# Scatter plot for P_train data
axs[0, 1].scatter(trainlabel[:, 1], best_y_train.iloc[:, 1])
axs[0, 1].set_xlabel('P_train_ture')
axs[0, 1].set_ylabel('P_train_pred')
axs[0, 1].set_title('P_train_ture & P_train_pred')

# Calculate and annotate R^2 for P_train data
r2_train_p = r2_score(trainlabel[:, 1], best_y_train.iloc[:, 1])
axs[0, 1].annotate(f'R^2 = {r2_train_p:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

# Scatter plot for test data
axs[1, 0].scatter(testlabel[:, 0], y_test.iloc[:, 0])
axs[1, 0].set_xlabel('TCO_test_ture')
axs[1, 0].set_ylabel('TCO_test_pred')
axs[1, 0].set_title('TCO_test_ture & TCO_test_pred')

# Calculate and annotate R^2 for test data
r2_test_tco = r2_score(testlabel[:, 0], y_test.iloc[:, 0])
axs[1, 0].annotate(f'R^2 = {r2_test_tco:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

# Scatter plot for P_test data
axs[1, 1].scatter(testlabel[:, 1], y_test.iloc[:, 1])
axs[1, 1].set_xlabel('P_test_ture')
axs[1, 1].set_ylabel('P_test_pred')
axs[1, 1].set_title('P_test_ture & P_test_pred')

# Calculate and annotate R^2 for P_test data
r2_test_p = r2_score(testlabel[:, 1], y_test.iloc[:, 1])
axs[1, 1].annotate(f'R^2 = {r2_test_p:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

# Create the fifth image and display it below
ax = plt.subplot(313)
# Create a new subgraph in the third row, located lower than the subgraph in the third row
ax.scatter(dataset2832[0:360, 8], COP_train_true[0:360], label='COP_train_true_7_28', color='grey')
ax.scatter(dataset2832[360:720, 8], COP_train_true[360:720], label='COP_train_true_7_32', color='grey')
ax.scatter(dataset7_30[:, 8], COP_test_true, label='COP_test_true_7_30', color='grey')
ax.scatter(dataset2832[0:360, 8], COP_train.iloc[0:360, 0], label='COP_train_7_28', color='#336699')
ax.scatter(dataset2832[360:720, 8], COP_train.iloc[360:720, 0], label='COP_train_7_32', color='#FFA500')
ax.scatter(dataset7_30[:, 8], COP_test.iloc[:, 0], label='COP_test_7_30', color='green')
ax.legend()  #336699 Deep Blue  #FFA500 Orange yellow  #228B22 Blue #800080 Purple
ax.set_xlabel('PLR')
ax.set_ylabel('COP')
ax.set_title('PLR --- COP')
plt.tight_layout()
# Adjust the layout of the subgraphs to fit the canvas size and fill in blank areas
# as much as possible to ensure that the spacing between subgraphs is appropriate
# and does not overlap with each other.
plt.show()
# endregion

# region Draw a training loss curve
print("Length of train_losses:", len(train_losses))
#for loss in train_losses:
#    print("Type:", type(loss), "Shape:", getattr(loss, 'shape', None))  # Check type and shape
plt.plot(train_losses, marker='o', linestyle='-')
plt.title('KPCBLS Training Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

train_losses = pd.DataFrame(train_losses)
train_losses.to_csv('train_losses_KPCBLS.csv')
# endregion

