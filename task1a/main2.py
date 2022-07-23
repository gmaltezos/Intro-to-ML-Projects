import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Importing the data
data = pd.read_csv('task1a/train.csv')

# lambdas for Ridge regression
ld = np.array([0.1, 1, 10, 100, 200])

# Initialization of solution array
solution = []

# Separating the X and Y data
y = data.iloc[:,0]
X = data.iloc[:,1:]

# Initializing the random splitting of the data in 10 folds
# cv = KFold(n_splits=10, shuffle = True, random_state = 0)
cv = KFold(n_splits=10)

# Iterating over the values of lambdas
for i in ld:

    # Temporary array for keeping the RMSE Error
    tempRMSE = []
    
    # Iterating over the 10 folds
    for train_index, test_index in cv.split(X):
        
        # Spliting the training and validation data
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        
        # Training the model using Ridge Regression
        clf = Ridge(alpha=i)
        clf.fit(X_train, y_train)
        
        # Calculating the RMS
        rms = mean_squared_error(y_test, clf.predict(X_test), squared=False)
        tempRMSE.append(rms)
        
    # Appending the mean RMSE to the solution array
    solution.append(np.mean(tempRMSE))
    
# Saving the solution to an output.csv file
pd.DataFrame(solution).to_csv("task1a/output2.csv", header=None, index=None)
