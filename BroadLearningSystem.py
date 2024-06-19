# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import time

# region Activation function, pseudo inverse function, loss function
def show_accuracy(predictLabel, Label): 
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))
def tansig(x):
    return (2/(1+np.exp(-2*x)))-1
def sigmoid(data):
    return 1.0/(1+np.exp(-data))
def linear(data):
    return data
def tanh(data):
    return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
def relu(data):
    return np.maximum(data, 0)
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

def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, dataset2832, dataset7_30, COP_train_true, COP_test_true):
    # region Training model
    time_start = time.time()  # Timing begins
    train_x = preprocessing.scale(train_x, axis=1)  # Standardize each row to have a mean of 0 and a standard deviation of 1, in order to eliminate dimensional differences between different eigenvalues
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])  # Choosing 0.1 here may be to ensure that the bias term does not have a significant impact on the model when the data is small, while also providing a certain degree of bias correction.
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    Beta1OfEachWindow = []
    distOfMaxAndMin = []
    minOfEachWindow = []
    for i in range(N2):
        random.seed(i)  # Generate different random numbers for each window
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1  # The purpose is to initialize the values of the weight matrix to a smaller range for faster training. Generally speaking, smaller initial weights can avoid gradient vanishing or exploding problems, and help the network converge to the appropriate solution faster.
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = FeatureOfInputDataWithBias.dot(betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3))-1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    # The purpose of orthogonalization is to ensure that the weight matrix has better numerical properties, such as reducing redundant information, avoiding gradient vanishing, etc., which helps with stable training and better performance.

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = sigmoid(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = np.dot(pinvOfInput,train_y)
    OutputOfTrain = np.array(np.dot(InputOfOutputLayer,OutputWeight))
    time_end = time.time()
    Training_time = time_end - time_start
    print('BLS Training has been finished!')
    print('The Total BLS Training Time is : ', round(Training_time, 6), ' seconds')
    # endregion

    # region Evaluation indicators for training results &  Store results
    # The evaluation indicators and training errors corresponding to the optimal iteration result
    mae_BLS_Train_TCO = mean_absolute_error(train_y[:, 0], OutputOfTrain[:, 0])  # calculate MAE_TCO
    mae_BLS_Train_P = mean_absolute_error(train_y[:, 1], OutputOfTrain[:, 1])  # calculate MAE_P
    rmse_BLS_Train_TCO = np.sqrt(mean_squared_error(train_y[:, 0], OutputOfTrain[:, 0]))  # calculate RMSE_TCO
    rmse_BLS_Train_P = np.sqrt(mean_squared_error(train_y[:, 1], OutputOfTrain[:, 1]))  # calculate RMSE_P
    r2_BLS_Train_TCO = r2_score(train_y[:, 0], OutputOfTrain[:, 0])  # calculate  R^2_TCO
    r2_BLS_Train_P = r2_score(train_y[:, 1], OutputOfTrain[:, 1])  # calculate R^2_P
    mape_BLS_Train_TCO = np.mean(
        np.abs((train_y[:, 0], OutputOfTrain[:, 0]) / train_y[:, 0])) * 100  # calculate MAPE_TCO
    mape_BLS_Train_P = np.mean(
        np.abs((train_y[:, 1], OutputOfTrain[:, 1]) / train_y[:, 1])) * 100  # calculate MAPE_TCO
    mse_BLS_Train_TCO = mean_squared_error(train_y[:, 0], OutputOfTrain[:, 0])  # calculate MSE_TCO
    mse_BLS_Train_P = mean_squared_error(train_y[:, 1], OutputOfTrain[:, 1])  # calculate MSE_P
    loss_0 = (6330 * dataset2832[:, 8].reshape(-1, 1) +
              OutputOfTrain[:, 1].reshape(-1, 1) -
              4.187 * train_x[:, 4].reshape(-1, 1) *
              (OutputOfTrain[:, 0].reshape(-1, 1) - train_x[:, 2].reshape(-1, 1)))
    loss_0 = np.where((loss_0 >= -760) & (loss_0 <= 760), 0.0,
                      np.where(loss_0 < -760, -760 - loss_0, loss_0 - 760))
    loss_energy_BLS_Train = np.mean(loss_0) / 2
    loss_BLS_Train = mse_BLS_Train_TCO + mse_BLS_Train_P + loss_energy_BLS_Train
    print('Training accuracy indicators')
    print("MAE BLS Train TCO:", mae_BLS_Train_TCO)
    print("MAE BLS Train P:", mae_BLS_Train_P)
    print("RMSE BLS Train TCO:", rmse_BLS_Train_TCO)
    print("RMSE BLS Train P:", rmse_BLS_Train_P)
    print("R^2 BLS Train TCO:", r2_BLS_Train_TCO)
    print("R^2 BLS Train P:", r2_BLS_Train_P)
    print("MAPE BLS Train TCO:", mape_BLS_Train_TCO)
    print("MAPE BLS Train P:", mape_BLS_Train_P)
    print("MSE BLS Train TCO:", mse_BLS_Train_TCO)
    print("MSE BLS Train P:", mse_BLS_Train_P)
    print("loss_energy BLS Train:", loss_energy_BLS_Train)
    print("loss BLS Train:", loss_BLS_Train)
    # dataset2832 is used in Scenario 1, and dataset283230fp is used in Scenario 2
    COP_train = 6330 * dataset2832[:, 8] / OutputOfTrain[:, 1]
    COP_train = pd.DataFrame(COP_train)
    # COP_train.to_csv('NETOUT_TRAIN_BLS_COP_fp_1.csv')
    y_train = pd.DataFrame(OutputOfTrain)
    #y_train.to_csv('NETOUT_TRAIN_BLS_TCO_P_fp_1.csv')
    print("Training results:")
    print(y_train)
    # endregion

    # region Testing model
    test_x = preprocessing.scale(test_x,axis = 1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] =(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = sigmoid(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)   # 这里有激活函数，原来是tansig
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
    OutputOfTest = np.array(np.dot(InputOfOutputLayerTest,OutputWeight))
    # endregion

    # region Evaluation indicators for testing results & Store results
    mae_BLS_Test_TCO = mean_absolute_error(test_y[:, 0], OutputOfTest[:, 0])
    mae_BLS_Test_P = mean_absolute_error(test_y[:, 1], OutputOfTest[:, 1])
    rmse_BLS_Test_TCO = np.sqrt(mean_squared_error(test_y[:, 0], OutputOfTest[:, 0]))
    rmse_BLS_Test_P = np.sqrt(mean_squared_error(test_y[:, 1], OutputOfTest[:, 1]))
    r2_BLS_Test_TCO = r2_score(test_y[:, 0], OutputOfTest[:, 0])
    r2_BLS_Test_P = r2_score(test_y[:, 1], OutputOfTest[:, 1])
    mape_BLS_Test_TCO = np.mean(np.abs((test_y[:, 0], OutputOfTest[:, 0]) / test_y[:, 0])) * 100
    mape_BLS_Test_P = np.mean(np.abs((test_y[:, 1], OutputOfTest[:, 1]) / test_y[:, 1])) * 100
    mse_BLS_Test_TCO = mean_squared_error(test_y[:, 0], OutputOfTest[:, 0])
    mse_BLS_Test_P = mean_squared_error(test_y[:, 1], OutputOfTest[:, 1])
    loss_0 = (6330 * dataset7_30[:, 8].reshape(-1, 1) +
              OutputOfTest[:, 1].reshape(-1, 1) -
              4.187 * dataset7_30[:, 4].reshape(-1, 1) *
              (OutputOfTest[:, 0].reshape(-1, 1) - dataset7_30[:, 2].reshape(-1, 1)))
    loss_0 = np.where((loss_0 >= -760) & (loss_0 <= 760), 0.0,
                      np.where(loss_0 < -760, -760 - loss_0, loss_0 - 760))
    loss_energy_BLS_Test = np.mean(loss_0) / 2
    loss_BLS_Test = mse_BLS_Test_TCO + mse_BLS_Test_P + loss_energy_BLS_Test
    error_Test = np.array([mae_BLS_Test_TCO,rmse_BLS_Test_TCO,r2_BLS_Test_TCO,
                                mae_BLS_Test_P,rmse_BLS_Test_P,r2_BLS_Test_P,
                                loss_energy_BLS_Test,Training_time]).reshape(1,-1)
    error_Test = pd.DataFrame(error_Test)
    error_Test.to_csv('NETOUT_TEST_BLS_error_fp_1.csv')
    print('Testing accuracy indicators')
    print("MAE BLS Test TCO:", mae_BLS_Test_TCO)
    print("MAE BLS Test P:", mae_BLS_Test_P)
    print("RMSE BLS Test TCO:", rmse_BLS_Test_TCO)
    print("RMSE BLS Test P:", rmse_BLS_Test_P)
    print("R^2 BLS Test TCO:", r2_BLS_Test_TCO)
    print("R^2 BLS Test P:", r2_BLS_Test_P)
    print("MAPE BLS Test TCO:", mape_BLS_Test_TCO)
    print("MAPE BLS Test P:", mape_BLS_Test_P)
    print("MSE BLS Test TCO:", mse_BLS_Test_TCO)
    print("MSE BLS Test P:", mse_BLS_Test_P)
    print("loss_energy BLS Test:", loss_energy_BLS_Test)
    print("loss BLS Test:", loss_BLS_Test)
    COP_test = 6330 * dataset7_30[:, 8] / OutputOfTest[:, 1]
    COP_test = pd.DataFrame(COP_test)
    #COP_test.to_csv('NETOUT_TEST_BLS_COP_fp_1.csv')
    y_pred = pd.DataFrame(OutputOfTest)
    #y_pred.to_csv('NETOUT_TEST_BLS_TCO_P_fp_1.csv')
    print("Testing results:")
    print(y_pred)
    # endregion

    # region Plotting
    # Create a figure and axis objects with a layout of (2, 2)
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    # Scatter plot for train data
    axs[0, 0].scatter(train_y[:, 0], y_train.iloc[:, 0])
    axs[0, 0].set_xlabel('TCO_train_ture')
    axs[0, 0].set_ylabel('TCO_train_pred')
    axs[0, 0].set_title('TCO_train_ture & TCO_train_pred')

    # Calculate and annotate R^2 for train data
    r2_train_tco = r2_score(train_y[:, 0], y_train.iloc[:, 0])
    axs[0, 0].annotate(f'R^2 = {r2_train_tco:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

    # Scatter plot for P_train data
    axs[0, 1].scatter(train_y[:, 1], y_train.iloc[:, 1])
    axs[0, 1].set_xlabel('P_train_ture')
    axs[0, 1].set_ylabel('P_train_pred')
    axs[0, 1].set_title('P_train_ture & P_train_pred')

    # Calculate and annotate R^2 for P_train data
    r2_train_p = r2_score(train_y[:, 1], y_train.iloc[:, 1])
    axs[0, 1].annotate(f'R^2 = {r2_train_p:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

    # Scatter plot for test data
    axs[1, 0].scatter(test_y[:, 0], y_pred.iloc[:, 0])
    axs[1, 0].set_xlabel('TCO_test_ture')
    axs[1, 0].set_ylabel('TCO_test_pred')
    axs[1, 0].set_title('TCO_test_ture & TCO_test_pred')

    # Calculate and annotate R^2 for test data
    r2_test_tco = r2_score(test_y[:, 0], y_pred.iloc[:, 0])
    axs[1, 0].annotate(f'R^2 = {r2_test_tco:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

    # Scatter plot for P_test data
    axs[1, 1].scatter(test_y[:, 1], y_pred.iloc[:, 1])
    axs[1, 1].set_xlabel('P_test_ture')
    axs[1, 1].set_ylabel('P_test_pred')
    axs[1, 1].set_title('P_test_ture & P_test_pred')

    # Calculate and annotate R^2 for P_test data
    r2_test_p = r2_score(test_y[:, 1], y_pred.iloc[:, 1])
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
    ax.legend()  # 336699 Deep Blue  #FFA500 Orange yellow  #228B22 Blue #800080 Purple
    ax.set_xlabel('PLR')
    ax.set_ylabel('COP')
    ax.set_title('PLR --- COP')
    plt.tight_layout()
    # Adjust the layout of the subgraphs to fit the canvas size and fill in blank areas
    # as much as possible to ensure that the spacing between subgraphs is appropriate
    # and does not overlap with each other.
    plt.show()
    # endregion

    return 0

'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数
M------步长
'''
def BLS_AddEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M,dataset2630, dataset6_28, COP_train_true, COP_test_true):
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    train_x = preprocessing.scale(train_x,axis = 1) #处理数据 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])

    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    Beta1OfEachWindow = []
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append( np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis =0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis =0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
 
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = sigmoid(tempOfOutputOfEnhanceLayer * parameterOfShrink)  # 这里有激活函数，原来是 tansig
    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = pinvOfInput.dot(train_y) 
    time_end=time.time() 
    trainTime = time_end - time_start
    
    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime
    
    test_x = preprocessing.scale(test_x, axis=1) 
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = sigmoid(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)  # 这里有激活函数，原来是 tansig

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
 
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time() #训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增量增加强化节点
    '''
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start=time.time()
        if N1*N2>= M : 
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M)-1)
        else :
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M).T-1).T
        
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s/np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = sigmoid(tempOfOutputOfEnhanceLayerAdd*parameterOfShrinkAdd[e])  # 这里有激活函数，原来是 tansig
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer,OutputOfEnhanceLayerAdd])
        
        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T,D)).I.dot(np.dot(D.T,pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e+1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        COP_train = 6330 * dataset2630[:, 8] / OutputOfTrain1[:, 1]
        COP_train = pd.DataFrame(COP_train)
        COP_train.to_csv('NETOUT_TRAIN_BLS_COP.csv')

        TrainingAccuracy = show_accuracy(OutputOfTrain1,train_y)
        train_acc[0][e+1] = TrainingAccuracy
        OutputOfTrain1 = pd.DataFrame(OutputOfTrain1)
        OutputOfTrain1.to_csv(f'NETOUT_TRAIN_BLS_INCRE_ENHANCE_{e}.csv',index=False)
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        print('增加增强节点后 训练结果')
        print(OutputOfTrain1)
        

        time_start = time.time()
        OutputOfEnhanceLayerAddTest = sigmoid(InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])  # 这里有激活函数，原来是 tansig
        InputOfOutputLayerTest=np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        COP_test = 6330 * dataset6_28[:, 8] / OutputOfTest1[:, 1]
        COP_test = pd.DataFrame(COP_test)
        COP_test.to_csv('NETOUT_TEST_BLS_COP.csv')
        TestingAcc = show_accuracy(OutputOfTest1,test_y)
        
        Test_time = time.time() - time_start
        test_time[0][e+1] = Test_time
        test_acc[0][e+1] = TestingAcc
        OutputOfTest1 = pd.DataFrame(OutputOfTest1)
        OutputOfTest1.to_csv(f'NETOUT_PRED_BLS_INCRE_ENHANCE_{e}.csv',index=False)
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %' )
        print('增加增强节点后 预测结果')
        print(OutputOfTest1)

    # region 画图
    # Create a figure and axis objects with a layout of (2, 2)
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    # Scatter plot for train data
    axs[0, 0].scatter(train_y[:, 0], OutputOfTrain1.iloc[:, 0])
    axs[0, 0].set_xlabel('TCO_train_ture')
    axs[0, 0].set_ylabel('TCO_train_pred')
    axs[0, 0].set_title('TCO_train_ture & TCO_train_pred')

    # Calculate and annotate R^2 for train data
    r2_train_tco = r2_score(train_y[:, 0], OutputOfTrain1.iloc[:, 0])
    axs[0, 0].annotate(f'R^2 = {r2_train_tco:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

    # Scatter plot for P_train data
    axs[0, 1].scatter(train_y[:, 1], OutputOfTrain1.iloc[:, 1])
    axs[0, 1].set_xlabel('P_train_ture')
    axs[0, 1].set_ylabel('P_train_pred')
    axs[0, 1].set_title('P_train_ture & P_train_pred')

    # Calculate and annotate R^2 for P_train data
    r2_train_p = r2_score(train_y[:, 1], OutputOfTrain1.iloc[:, 1])
    axs[0, 1].annotate(f'R^2 = {r2_train_p:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

    # Scatter plot for test data
    axs[1, 0].scatter(test_y[:, 0], OutputOfTest1.iloc[:, 0])
    axs[1, 0].set_xlabel('TCO_test_ture')
    axs[1, 0].set_ylabel('TCO_test_pred')
    axs[1, 0].set_title('TCO_test_ture & TCO_test_pred')

    # Calculate and annotate R^2 for test data
    r2_test_tco = r2_score(test_y[:, 0], OutputOfTest1.iloc[:, 0])
    axs[1, 0].annotate(f'R^2 = {r2_test_tco:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

    # Scatter plot for P_test data
    axs[1, 1].scatter(test_y[:, 1], OutputOfTest1.iloc[:, 1])
    axs[1, 1].set_xlabel('P_test_ture')
    axs[1, 1].set_ylabel('P_test_pred')
    axs[1, 1].set_title('P_test_ture & P_test_pred')

    # Calculate and annotate R^2 for P_test data
    r2_test_p = r2_score(test_y[:, 1], OutputOfTest1.iloc[:, 1])
    axs[1, 1].annotate(f'R^2 = {r2_test_p:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

    # 创建第五幅图并显示在下方
    ax = plt.subplot(313)  # 创建一个新的子图在第三行,比第三行的子图更下方

    ax.scatter(dataset2630[0:360, 8], COP_train_true[0:360], label='COP_train_true_6_26', color='grey')
    ax.scatter(dataset2630[360:720, 8], COP_train_true[360:720], label='COP_train_true_6_30', color='grey')
    ax.scatter(dataset6_28[:, 8], COP_test_true, label='COP_test_true_6_28', color='grey')
    ax.scatter(dataset2630[0:360, 8], COP_train.iloc[0:360, 0], label='COP_train_6_26', color='#336699')
    ax.scatter(dataset2630[360:720, 8], COP_train.iloc[360:720, 0], label='COP_train_6_30', color='#FFA500')
    ax.scatter(dataset6_28[:, 8], COP_test.iloc[:, 0], label='COP_test_6_28', color='green')
    ax.legend()  # 336699深蓝  #FFA500橙黄  #228B22也是蓝 #800080紫

    ax.set_xlabel('PLR')
    ax.set_ylabel('COP')
    ax.set_title('PLR --- COP')
    plt.tight_layout()
    plt.show()
    # endregion
        
    return test_acc,test_time,train_acc,train_time

'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
L------步数

M1-----增加映射节点数
M2-----与增加映射节点对应的强化节点数
M3-----新增加的强化节点
'''
def BLS_AddFeatureEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M1,M2,M3):
    
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    train_x = preprocessing.scale(train_x,axis = 1) 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])

    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis = 0) - np.min(outputOfEachWindow,axis = 0))
        minOfEachWindow.append(np.mean(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
 
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain,c)
    OutputWeight =pinvOfInput.dot(train_y) #全局违逆
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    OutputOfTrain = np.dot(InputOfOutputLayerTrain,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    test_x = preprocessing.scale(test_x,axis = 1) 
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i] 

    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
  
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time() 
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增加Mapping 和 强化节点
    '''
    WeightOfNewFeature2 = list()
    WeightOfNewFeature3 = list()
    for e in list(range(L)):
        time_start = time.time()
        random.seed(e+N2+u)
        weightOfNewMapping = 2 * random.random([train_x.shape[1]+1,M1]) - 1
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)

        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)
        betaOfNewWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfNewWindow)
   
        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)
        distOfMaxAndMin.append( np.max(TempOfFeatureOutput,axis = 0) - np.min(TempOfFeatureOutput,axis = 0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput,axis = 0))
        outputOfNewWindow = (TempOfFeatureOutput-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e]

        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer,outputOfNewWindow])

        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0],1))])

        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2])-1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2]).T-1).T  
        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)
        
        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)
        
        parameter1 = s/np.max(tempOfNewFeatureEhanceNodes)

        outputOfNewFeatureEhanceNodes = tansig(tempOfNewFeatureEhanceNodes * parameter1)

        if N2*N1+e*M1>=M3:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3) - 1)
        else:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3).T-1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)

        InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s/np.max(tempOfNewEnhanceNodes)
        OutputOfNewEnhanceNodes = tansig(tempOfNewEnhanceNodes * parameter2)
        OutputOfTotalNewAddNodes = np.hstack([outputOfNewWindow,outputOfNewFeatureEhanceNodes,OutputOfNewEnhanceNodes])
        tempOfInputOfLastLayes = np.hstack([InputOfOutputLayerTrain,OutputOfTotalNewAddNodes])
        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)
        
        if C.all() == 0:
            w = D.shape[1]
            B = (np.eye(w)- D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeight = pinvOfInput.dot(train_y)        
        InputOfOutputLayerTrain = tempOfInputOfLastLayes
        
        time_end = time.time()
        Train_time = time_end - time_start
        train_time[0][e+1] = Train_time
        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        TrainingAccuracy = show_accuracy(predictLabel,train_y)
        train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        
        # 测试过程
        #先生成新映射窗口输出
        time_start = time.time() 
        WeightOfNewMapping =  Beta1OfEachWindow[N2+e]

        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping )
        
        outputOfNewWindowTest = (outputOfNewWindowTest-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e] 
        
        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,outputOfNewWindowTest])
        
        InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest,0.1*np.ones([OutputOfFeatureMappingLayerTest.shape[0],1])])
        
        NewInputOfEnhanceLayerWithBiasTest = np.hstack([outputOfNewWindowTest,0.1*np.ones([outputOfNewWindowTest.shape[0],1])])

        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]
        
        OutputOfRelateEnhanceNodes = tansig(NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)
        
        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]
        
        OutputOfNewEnhanceNodes = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes)*parameter2)
        
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest,outputOfNewWindowTest,OutputOfRelateEnhanceNodes,OutputOfNewEnhanceNodes])
    
        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)

        TestingAccuracy = show_accuracy(predictLabel,test_y)
        time_end = time.time()
        Testing_time= time_end - time_start
        test_time[0][e+1] = Testing_time
        test_acc[0][e+1]=TestingAccuracy
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %' )

    return test_acc,test_time,train_acc,train_time

