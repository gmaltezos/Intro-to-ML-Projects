import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

# Importing the data
dataX = pd.read_csv('task2/train_features.csv')
dataY = pd.read_csv('task2/train_labels.csv')
testData = pd.read_csv('task2/test_features.csv')

# # Drop Columns
# dataX = dataX.drop(['EtCO2', 'Fibrinogen', 'Bilirubin_direct', 'TroponinI'], axis = 1)
# testData = testData.drop(['EtCO2', 'Fibrinogen', 'Bilirubin_direct', 'TroponinI'], axis = 1)

# Isolating the numerical data 
rowID = dataX.iloc[:, 0]
dataX = dataX.groupby(['pid']).mean()
testData = testData.groupby(['pid']).mean()
dataX = dataX.drop('Time', axis=1, inplace=False)
testData = testData.drop('Time', axis=1, inplace=False)

# Data Sorting
dataX = dataX.sort_values(by=['pid'])
dataY = dataY.sort_values(by=['pid'])
dataY = dataY.drop('pid', axis=1, inplace=False)

# Imputing the missing values using K-Nearest Neighbour
imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
imp = imputer.fit(dataX)
X_train_imp = imp.transform(dataX)
X_test_imp = imp.transform(testData)

# Resetting the indices
dataX = dataX.reset_index()
testData = testData.reset_index()

# Data Scaling
scaler = StandardScaler()
X_train_imp = scaler.fit_transform(X_train_imp)
X_test_imp = scaler.fit_transform(X_test_imp)

# Instantiating the output dataframe
output = pd.DataFrame({'pid': testData['pid']})

# SUB-TASK 1: ORDERING OF MEDICAL TEST and SUB-TASK 2: SEPSIS PREDICTION
labelsOfInterest = ["LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", "LABEL_EtCO2", "LABEL_Sepsis"]

# Iterating over the labels for Classification
for label in labelsOfInterest:
    
    # Training the model 
    xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 100, eval_metric = 'auc')
    xg_reg.fit(X_train_imp,dataY[label])

    # Adding the predictions on the test data to the output dataframe
    predictions = xg_reg.predict(X_test_imp)
    output[label] = predictions


# SUB-TASK 3: KEYS VITALS SIGNS PREDICTIONS
labelsRegression = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]

# Perform ridge regression with cross-validation for the regression labels
ld = np.array([0.01, 0.1, 1, 10, 100, 200])

for label in labelsRegression:
    reg = LassoCV(alphas=ld,cv=10,selection='random',fit_intercept=True,random_state=42).fit(X_train_imp, dataY[label])
    # reg = RidgeCV().fit(X_train_imp, dataY[label])
    # kr = KernelRidge(alpha = 1, kernel='polynomial', degree = 3)
    # reg = kr.fit(X_train_imp, dataY[label])
    outputReg = reg.predict(X_test_imp)
    output[label] = outputReg

# Saving the results
output.to_csv('task2/output.zip', encoding='utf-8', index=False, float_format='%.3f', compression='zip')