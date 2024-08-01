"""
Demo_MLP -

Author: 
Date: 2024-05-04
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# region Define MLP structure
def create_model():
    model = Sequential()  # Creating a multi-layer perceptron model
    model.add(Dense(100, input_dim=traindata.shape[1], activation='relu'))  # Add input layer and first hidden layer (100 neurons)
    model.add(Dense(300, activation='relu'))  # Add a second hidden layer (300 neurons) relu linear
    model.add(Dense(trainlabel.shape[1]))  # Add output layer sigmoid tanh (No, the effect is very poor)
    model.compile(loss='mean_squared_error', optimizer='adam')  # Compile model
    return model
# In Keras, Sequential is a class used to construct neural network models.
# It allows the addition of neural network layers in order to construct various types of neural networks,
# including multi-layer perceptrons (MLP), convolutional neural networks (CNN), recurrent neural networks (RNN), and so on.
# First import the Sequential class, and then create a Sequential object called model to build a neural network model.
# Next, use the model. add() method to add neural network layers layer by layer to construct the model structure.
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

# region Training & Testing model
time_start = time.time()  # Timing begins
seed_value = 42
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
num_runs = 1   # Multiple training sessions
max_epochs = 700
results_train = []  # Create a list to store the results of multiple runs, and finally take the average of multiple trainings as the final training result
results_test = []
train_losses = []   # Store training losses
for _ in range(num_runs):
    model = create_model()  #Create model instance
    history = model.fit(traindata, trainlabel, epochs=max_epochs, batch_size=60)  # Training model, verbose=0, the model will not output any information during the training process
    train_losses.append(history.history['loss'])    # 记录训练损失
    result_train = model.predict(traindata)
    results_train.append(result_train)
    result_test = model.predict(testdata)
    results_test.append(result_test)
y_train = np.mean(results_train, axis=0)  # Calculate the average value along the first dimension of the result list (sample dimension)
y_pred = np.mean(results_test, axis=0)
Training_time = time.time() - time_start  # End timing
print('MLP Training has been finished!')
print('The Total MLP Training Time is : ', round(Training_time, 6), ' seconds')
# endregion

# region Evaluation indicators for training results &  Store results
# The evaluation indicators and training errors corresponding to the optimal iteration result
mae_MLP_Train_TCO = mean_absolute_error(trainlabel[:, 0], y_train[:, 0])  # calculate MAE_TCO
mae_MLP_Train_P = mean_absolute_error(trainlabel[:, 1], y_train[:, 1])  # calculate MAE_P
rmse_MLP_Train_TCO = np.sqrt(mean_squared_error(trainlabel[:, 0], y_train[:, 0]))  # calculate RMSE_TCO
rmse_MLP_Train_P = np.sqrt(mean_squared_error(trainlabel[:, 1], y_train[:, 1]))  # calculate RMSE_P
r2_MLP_Train_TCO = r2_score(trainlabel[:, 0], y_train[:, 0])  # calculate  R^2_TCO
r2_MLP_Train_P = r2_score(trainlabel[:, 1], y_train[:, 1])  # calculate R^2_P
mape_MLP_Train_TCO = np.mean(np.abs((trainlabel[:,0], y_train[:,0]) / trainlabel[:,0])) * 100  # calculate MAPE_TCO
mape_MLP_Train_P = np.mean(np.abs((trainlabel[:,1], y_train[:,1]) / trainlabel[:,1])) * 100  # calculate MAPE_TCO
mse_MLP_Train_TCO = mean_squared_error(trainlabel[:, 0], y_train[:, 0])  # calculate MSE_TCO
mse_MLP_Train_P = mean_squared_error(trainlabel[:, 1], y_train[:, 1])  # calculate MSE_P
# dataset2832 is used in Scenario 1, and dataset283230fp is used in Scenario 2
loss_0 = (6330 * dataset2832[:, 8].reshape(-1, 1) +
          y_train[:, 1].reshape(-1, 1) -
          4.187 * traindata[:, 4].reshape(-1, 1) *
          (y_train[:, 0].reshape(-1, 1) - traindata[:, 2].reshape(-1, 1)))
loss_0 = np.where((loss_0 >= -760) & (loss_0 <= 760), 0.0,
                  np.where(loss_0 < -760, -760 - loss_0, loss_0 - 760))
loss_energy_MLP_Train = np.mean(loss_0) / 2
loss_MLP_Train = mse_MLP_Train_TCO + mse_MLP_Train_P + loss_energy_MLP_Train
print('Training accuracy indicators')
print("MAE MLP Train TCO:", mae_MLP_Train_TCO)
print("MAE MLP Train P:", mae_MLP_Train_P)
print("RMSE MLP Train TCO:", rmse_MLP_Train_TCO)
print("RMSE MLP Train P:", rmse_MLP_Train_P)
print("R^2 MLP Train TCO:", r2_MLP_Train_TCO)
print("R^2 MLP Train P:", r2_MLP_Train_P)
print("MAPE MLP Train TCO:", mape_MLP_Train_TCO)
print("MAPE MLP Train P:", mape_MLP_Train_P)
print("MSE MLP Train TCO:", mse_MLP_Train_TCO)
print("MSE MLP Train P:", mse_MLP_Train_P)
print("loss_energy MLP Train:", loss_energy_MLP_Train)
print("loss MLP Train:", loss_MLP_Train)
COP_train = 6330 * dataset2832[:, 8] / y_train[:, 1]
COP_train = pd.DataFrame(COP_train)
#COP_train.to_csv('NETOUT_TRAIN_MLP_COP_fp_1.csv')
best_y_train = pd.DataFrame(y_train)
#best_y_train.to_csv('NETOUT_TRAIN_MLP_TCO_P_fp_1.csv')
print("Training results:")
print(best_y_train)
# endregion

# region Evaluation indicators for testing results & Store results
mae_MLP_Test_TCO = mean_absolute_error(testlabel[:,0], y_pred[:,0])
mae_MLP_Test_P = mean_absolute_error(testlabel[:,1], y_pred[:,1])
rmse_MLP_Test_TCO = np.sqrt(mean_squared_error(testlabel[:,0], y_pred[:,0]))
rmse_MLP_Test_P = np.sqrt(mean_squared_error(testlabel[:,1], y_pred[:,1]))
r2_MLP_Test_TCO = r2_score(testlabel[:,0], y_pred[:,0])
r2_MLP_Test_P = r2_score(testlabel[:,1], y_pred[:,1])
mape_MLP_Test_TCO = np.mean(np.abs((testlabel[:,0], y_pred[:,0]) / testlabel[:,0])) * 100
mape_MLP_Test_P = np.mean(np.abs((testlabel[:,1], y_pred[:,1]) / testlabel[:,1])) * 100
mse_MLP_Test_TCO = mean_squared_error(testlabel[:, 0], y_pred[:, 0])
mse_MLP_Test_P = mean_squared_error(testlabel[:, 1], y_pred[:, 1])
loss_2 = (6330 * dataset7_30[:, 8].reshape(-1, 1) +
          y_pred[:, 1].reshape(-1, 1) -
          4.187 * testdata[:, 4].reshape(-1, 1) *
          (y_pred[:, 0].reshape(-1, 1) - testdata[:, 2].reshape(-1, 1)))
loss_2 = np.where((loss_2 >= -760) & (loss_2 <= 760), 0.0,
                  np.where(loss_2 < -760, -760 - loss_2, loss_2 - 760))
loss_energy_MLP_Test = np.mean(loss_2) / 2
loss_MLP_Test = mse_MLP_Test_TCO + mse_MLP_Test_P + loss_energy_MLP_Test
error_Test = np.array([mae_MLP_Test_TCO, rmse_MLP_Test_TCO, r2_MLP_Test_TCO,
                       mae_MLP_Test_P, rmse_MLP_Test_P, r2_MLP_Test_P,
                       loss_energy_MLP_Test, Training_time]).reshape(1, -1)
error_Test = pd.DataFrame(error_Test)
error_Test.to_csv('NETOUT_TEST_MLP_error_fp_1.csv')
print("Testing accuracy indicators")
print("MAE MLP Test TCO:", mae_MLP_Test_TCO)
print("MAE MLP Test P:", mae_MLP_Test_P)
print("RMSE MLP Test TCO:", rmse_MLP_Test_TCO)
print("RMSE MLP Test P:", rmse_MLP_Test_P)
print("R^2 MLP Test TCO:", r2_MLP_Test_TCO)
print("R^2 MLP Test P:", r2_MLP_Test_P)
print("MAPE MLP Test TCO:", mape_MLP_Test_TCO)
print("MAPE MLP Test P:", mape_MLP_Test_P)
print("MSE MLP Test TCO:", mse_MLP_Test_TCO)
print("MSE MLP Test P:", mse_MLP_Test_P)
print("loss_energy MLP Test:", loss_energy_MLP_Test)
print("loss MLP Test:", loss_MLP_Test)
COP_test = 6330 * dataset7_30[:, 8] / y_pred[:, 1]
COP_test = pd.DataFrame(COP_test)
#COP_test.to_csv('NETOUT_TEST_MLP_COP_fp_1.csv')
y_pred = pd.DataFrame(y_pred)
#y_pred.to_csv('NETOUT_TEST_MLP_TCO_P_fp_1.csv')
print("Testing results:")
print(y_pred)
# endregion

# region Plotting experimental results
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
axs[1, 0].scatter(testlabel[:, 0], y_pred.iloc[:, 0])
axs[1, 0].set_xlabel('TCO_test_ture')
axs[1, 0].set_ylabel('TCO_test_pred')
axs[1, 0].set_title('TCO_test_ture & TCO_test_pred')

# Calculate and annotate R^2 for test data
r2_test_tco = r2_score(testlabel[:, 0], y_pred.iloc[:, 0])
axs[1, 0].annotate(f'R^2 = {r2_test_tco:.6f}', xy=(0.1, 0.9), xycoords='axes fraction')

# Scatter plot for P_test data
axs[1, 1].scatter(testlabel[:, 1], y_pred.iloc[:, 1])
axs[1, 1].set_xlabel('P_test_ture')
axs[1, 1].set_ylabel('P_test_pred')
axs[1, 1].set_title('P_test_ture & P_test_pred')

# Calculate and annotate R^2 for P_test data
r2_test_p = r2_score(testlabel[:, 1], y_pred.iloc[:, 1])
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
plt.plot(np.arange(1, max_epochs + 1), np.mean(train_losses, axis=0))
plt.title('MLP_Training Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# endregion

# 使用对数坐标绘制训练损失曲线
#plt.figure(figsize=(10, 6))
#plt.semilogy(np.arange(1, 1001), np.mean(train_losses, axis=0), label='Training Loss')
#plt.title('Training Loss vs. Epochs (Log Scale)')
#plt.xlabel('Epochs')
#plt.ylabel('Loss (Log Scale)')
#plt.legend()
#plt.grid()
#plt.show()


train_losses = pd.DataFrame(train_losses)
#train_losses.to_csv('train_losses_MLP.csv')

