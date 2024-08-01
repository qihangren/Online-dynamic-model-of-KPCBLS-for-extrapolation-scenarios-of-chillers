"""
Demo_KPCMLP -

Author: 
Date: 2024-05-15
"""

import time
import numpy as np
import pandas as pd
from tensorflow.keras import layers, Model
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# region Custom loss function, Define MLP structure, Define early stop
class CustomLossWrapper:
    def custom_loss(self, batch_size):
        def loss(y_true, y_pred):
            sliced_traindata = traindata[:batch_size, :]  # Slice training data based on batch size
            # Here, the loss can be calculated based on the input data sliced_traindata and the predicted value y_pred of the model
            mse_TCO = tf.reduce_mean(tf.square(y_true[:,0] - y_pred[:,0]))
            mse_P = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))
            loss_en = (
                4.187 * sliced_traindata[:, 3] * (sliced_traindata[:, 1] - sliced_traindata[:, 0]) +
                y_pred[:, 1] -
                4.187 * sliced_traindata[:, 4] * (y_pred[:, 0] - sliced_traindata[:, 2])
            )
            condition1 = tf.logical_or(loss_en >= -760, loss_en <= 760)
            condition2 = loss_en < -760
            condition3 = loss_en > 760
            loss_en = tf.where(condition1, tf.zeros_like(loss_en), loss_en)
            loss_en = tf.where(condition2, -760 - loss_en, loss_en)
            loss_en = tf.where(condition3, loss_en - 760, loss_en)
            loss_en = tf.reduce_mean(loss_en)
            return mse_TCO + mse_P + loss_en
        return loss
class MLP(Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = layers.Dense(100, activation='relu')
        self.dense2 = layers.Dense(300, activation='relu')
        self.dense3 = layers.Dense(2, activation='linear')  # Output layer, with an output dimension of 2

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
# Custom callback function to stop training when the loss value reaches or falls below the specified value
class EarlyStoppingAtLoss(Callback):
    def __init__(self, monitor='loss', value=0, verbose=0):
        super(EarlyStoppingAtLoss, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.best = float('inf')
        self.losses = []  # To store losses over epochs

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        self.losses.append(current)  # Record the loss value

        # Check if current loss is lower than the specified threshold value
        if current < self.value:
            if self.verbose > 0:
                print(f'\nEpoch {epoch + 1}: early stopping, loss < {self.value}')
            self.model.stop_training = True
        else:
            # Update best loss seen so far
            if current < self.best:
                self.best = current

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
#  Defined a "CustomimLossWrapper" class that takes training data "traindata" as a parameter.
#  In the "CustomiLossWrapper" class, a "custom_loss" method is defined as the loss function.
#  This loss function can access "traindata" data and perform reduction operations according to batch size in each batch.
#  Finally, an instance of "CustomiLossWrapper" was created and the "custom_loss" method was passed during model compilation.
#  This ensures that the loss function is calculated correctly and that the training data is accessible.
seed_value = 42
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
num_runs = 1   # Multiple training sessions
results_train = []  # Create a list to store the results of multiple runs, and finally take the average of multiple trainings as the final training result
results_test = []
for _ in range(num_runs):
    batch_size = 60  # Ensure that the sample size is 720 and can be evenly divided by the batch size, 15,30,45,60,90.Scenario one, choose 60, Scenario two, for every additional 36 samples+3,63--90
    model = MLP()  # Define model
    loss_wrapper = CustomLossWrapper()  # Create an instance of a loss function wrapper
    model.compile(optimizer='adam',
                  loss=loss_wrapper.custom_loss(batch_size))  # When compiling the model, pass the loss function in the loss function wrapper and pass the batch size "batch_size"
    early_stopping = EarlyStoppingAtLoss(monitor='loss', verbose=1)
    early_stopping.value = 90
    model.fit(traindata, trainlabel, epochs=700, batch_size=batch_size,callbacks=[early_stopping])  # Training model
    result_train = model.predict(traindata)
    results_train.append(result_train)
    result_test = model.predict(testdata)
    results_test.append(result_test)
y_train = np.mean(results_train,axis=0)  # Calculate the average value along the first dimension of the result list (sample dimension)
y_pred = np.mean(results_test,axis=0)
Training_time = time.time() - time_start  # End timing
print('KPCMLP Training has been finished!')
print('The Total KPCMLP Training Time is : ', round(Training_time, 6), ' seconds')
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
print("MAE KPCMLP Train TCO:", mae_MLP_Train_TCO)
print("MAE KPCMLP Train P:", mae_MLP_Train_P)
print("RMSE KPCMLP Train TCO:", rmse_MLP_Train_TCO)
print("RMSE KPCMLP Train P:", rmse_MLP_Train_P)
print("R^2 KPCMLP Train TCO:", r2_MLP_Train_TCO)
print("R^2 KPCMLP Train P:", r2_MLP_Train_P)
print("MAPE KPCMLP Train TCO:", mape_MLP_Train_TCO)
print("MAPE KPCMLP Train P:", mape_MLP_Train_P)
print("MSE KPCMLP Train TCO:", mse_MLP_Train_TCO)
print("MSE KPCMLP Train P:", mse_MLP_Train_P)
print("loss_energy KPCMLP Train:", loss_energy_MLP_Train)
print("loss KPCMLP Train:", loss_MLP_Train)
COP_train = 6330 * dataset2832[:, 8] / y_train[:, 1]
COP_train = pd.DataFrame(COP_train)
#COP_train.to_csv('NETOUT_TRAIN_KPCMLP_COP_fp_1.csv')
best_y_train = pd.DataFrame(y_train)
#best_y_train.to_csv('NETOUT_TRAIN_KPCMLP_TCO_P_fp_1.csv')
print("Training results:")
print(best_y_train)
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
y_pred = 0.5 * y_pred + 0.5 * y_knowledge
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
error_Test.to_csv('NETOUT_TEST_KPCMLP_error_fp_1.csv')
print("Testing accuracy indicators")
print("MAE KPCMLP Test TCO:", mae_MLP_Test_TCO)
print("MAE KPCMLP Test P:", mae_MLP_Test_P)
print("RMSE KPCMLP Test TCO:", rmse_MLP_Test_TCO)
print("RMSE KPCMLP Test P:", rmse_MLP_Test_P)
print("R^2 KPCMLP Test TCO:", r2_MLP_Test_TCO)
print("R^2 KPCMLP Test P:", r2_MLP_Test_P)
print("MAPE KPCMLP Test TCO:", mape_MLP_Test_TCO)
print("MAPE KPCMLP Test P:", mape_MLP_Test_P)
print("MSE KPCMLP Test TCO:", mse_MLP_Test_TCO)
print("MSE KPCMLP Test P:", mse_MLP_Test_P)
print("loss_energy KPCMLP Test:", loss_energy_MLP_Test)
print("loss KPCMLP Test:", loss_MLP_Test)
COP_test = 6330 * dataset7_30[:, 8] / y_pred[:, 1]
COP_test = pd.DataFrame(COP_test)
#COP_test.to_csv('NETOUT_TEST_KPCMLP_COP_fp_1.csv')
y_pred = pd.DataFrame(y_pred)
#y_pred.to_csv('NETOUT_TEST_KPCMLP_TCO_P_fp_1.csv')
print("Testing results:")
print(y_pred)
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

# Assuming `early_stopping` is an instance of `EarlyStoppingAtLoss`
train_losses = early_stopping.losses
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, marker='o', label='Training Loss')
plt.title('KPCMLP Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

train_losses = pd.DataFrame(train_losses)
train_losses.to_csv('train_losses_KPCMLP.csv')
