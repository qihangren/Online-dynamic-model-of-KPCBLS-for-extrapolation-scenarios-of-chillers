# An online dynamic model based on Knowledge-plus-Physical-Constraint Broad Learning System for extrapolation scenarios of chillers
## Abstract:
The performance change of chiller under different operating conditions is an important factor that needs to be considered in the accurate formulation of energy-saving control strategy for central air conditioning system. In the Dynamic Data-Driven Application System framework, the Knowledge-plus-Physical-Constraint Broad Learning System (KPCBLS) method is used to realize online extrapolation of unknown working conditions of chillers. The core idea of the KPCBLS method is to ensure that the output of the model under unknown working conditions conforms to the laws of physics. Through data experiments, KPCBLS was compared with Knowledge-plus-Physical-Constraint Multi-Layer Perceptron. The experimental results showed that the mean absolute error (MAE) and root mean squared error (RMSE) of the former were reduced by 50.78% and 52.50% respectively compared to the latter, while the coefficient of determination (R<sup>2</sup>) was increased by 19.91%, and the training time was shortened by nearly 98.23%. The method proposed in this article can balance the complexity of model structure, model accuracy, and model training time complexity in the extrapolation scenario modeling of chillers.

## Datasets:
The data in this study were obtained on a chiller with a rated cooling capacity of 6330kw, a rated power of 1194kw, a rated COP of 5.303, a chilled water constant flow (Gew) of 251.6L/s and a cooling water constant flow (Gcw) of 361.5L/s. 

1. Working condition 1: T<sub>ew,L</sub>=7℃，T<sub>cw,E</sub>=28℃，PLR=30%~100%，360 sampling points. Corresponding to "7-28.csv" sample data file.
2. Working condition 2: T<sub>ew,L</sub>=7℃，T<sub>cw,E</sub>=30℃，PLR=30%~100%，360 sampling points. Corresponding to "7-30.csv" sample data file.
3. Working condition 3: T<sub>ew,L</sub>=7℃，T<sub>cw,E</sub>=32℃，PLR=30%~100%，360 sampling points. Corresponding to "7-32.csv" sample data file.

### Dependencies:

* Python 3.6.13
* numpy 1.19.5
* pandas 1.1.5
* scipy 1.5.4
* matplotlib 3.3.4
* scikit-learn 0.24.2
* tensorflow 1.15.0

## Using the code
1. Six models：BLS(Demo_BLS.py), PCBLS(Demo_PCBLS.py), KPCBLS(Demo_KPCBLS.py), MLP(Demo_MLP.py), PCMLP(Demo_PCMLP.py), KPCMLP(Demo_KPCMLP.py)
2. Two scenarios: Extrapolation scenario, Online scenarios. The scene switching is achieved by changing the training set data. Perform performance tests on six models using two scenarios.

### The structure of the code
Each model program consists of the following parts:
1. Import necessary library files
2. Custom functions
3. Dataset processing
4. Training model
5. Testing model
6. Evaluation indicators for training and testing results
7. Store results
8. Plotting

### Taking the KPCBLS model as an example for explanation
Here is only a brief explanation. For detailed program instructions, please refer to the comments in the program.<br>
When switching scenes, relevant operations need to be performed in the program:
1. Change the values of the following variables in **Dataset processing**
* traindata
* trainlabel
* COP_train_true
2. Change the values of the following variables in **Evaluation indicators for model training results and model testing results**
* Ensure the correct training set is used for calculating **loss** and **COP**
