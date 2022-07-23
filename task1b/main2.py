import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold

# Importing the data
data = pd.read_csv('task1b/train.csv')

# Initialization of temporary array to hold coefficients for each iteration
# tempCoeffients = np.array([[0]*21 for i in range(10)])
# column_names = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20", "c21"]
tempCoeffients = pd.DataFrame()

# Separating the X and Y data
Y = data.iloc[:,1]
X = data.iloc[:,2:]

# Expand the Data matrix, to include the transformations needed for the products with the weights
X_expanded = np.concatenate([X, np.power(X,2), np.exp(X), np.cos(X), np.ones((700,1))], axis=1)

# Initializing the random splitting of the data in 10 folds
cv = KFold(n_splits=10, shuffle=True, random_state=10)

# Defining a lamda array to train with
ld = np.array([0.1, 1, 10, 100, 200])

# Initialising a loop counter
loopCounter = 0

# Iterating over the 10 folds
for train_index, test_index in cv.split(X_expanded):

    # Spliting the training and validation data
    X_train, X_test, Y_train, Y_test = X_expanded[train_index], X_expanded[test_index], Y[train_index], Y[test_index]
   
    # Training the model using Lasso Regression
    reg = Lasso(alphas=ld, selection='random', fit_intercept=False, random_state=42).fit(X_train, Y_train)
    
    # Attaining the coefficients
    coefs = reg.coef_

    # Storing the coefficients of this training set
    tempCoeffients = pd.concat([tempCoeffients, pd.DataFrame(coefs)], axis=1)

    # Increment the loop counter
    loopCounter += 1

# print(tempCoeffients)

# Take average of columns for coefficients
meanCoefficients = tempCoeffients.mean(axis=1)

# Saving the solution to an output.csv file
pd.DataFrame(meanCoefficients).to_csv("task1b/output2.csv", header=None, index=None)