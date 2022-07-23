import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, train_test_split

# Importing the data
data = pd.read_csv('task1a/train.csv')

# lambdas for Ridge regression
ld = np.array([0.1, 1, 10, 100, 200])

# Initialization of solution array
solution = []


# y = data.iloc[:,0]
# X = data.iloc[:,1:]
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# model = RidgeCV(alphas=[1], cv=cv, scoring='neg_mean_absolute_error')
# model.fit(data.iloc[0:134, 1:], data.iloc[0:134, 0:1])
# rms = mean_squared_error(data.iloc[135:149, 0:1], model.predict(data.iloc[135:149, 1:]), squared=False)
# solution.append(rms)

# Iterating over the 5 different lambdas
for i in ld:
    # Temporary array for keeping the RMSE Error
    tempRMSE = []

    # Split the dataset into 10 folds and hold one out during every iteration for validation (RSME computation)
    for k in range(10):

        # Isolating the training and validation data
        trainingData = pd.concat([data.iloc[0:int((len(data)/10)*k),:], data.iloc[int((len(data)/10)*(k+1)):,:]])
        validationData = data.iloc[int((len(data)/10)*k):int((len(data)/10)*(k+1))]

        # Separating the X and Y data
        trainingDataY = trainingData.iloc[:,0]
        validationDataY = validationData.iloc[:,0]
        trainingDataX = trainingData.iloc[:,1:]
        validationDataX = validationData.iloc[:,1:]
        
        # Training the model using Ridge Regression
        clf = Ridge(alpha=i)
        clf.fit(trainingDataX, trainingDataY)
        
        # Calculating the RMSE
        rms = mean_squared_error(validationDataY, clf.predict(validationDataX), squared=False)
        tempRMSE.append(rms)

    # Appending the mean RMSE to the solution array
    solution.append(np.mean(tempRMSE))

# Saving the solution to an output.csv file
pd.DataFrame(solution).to_csv("task1a/output.csv", header=None, index=None)

