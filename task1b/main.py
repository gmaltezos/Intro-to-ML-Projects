import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV

# Importing the data
data = pd.read_csv('task1b/train.csv')

# Split into input and outcomes
y = data.iloc[:,1]
X = data.iloc[:,2:]

# Expand the Data matrix, to include the transformations needed for the products with the weights
X_expanded = np.concatenate([X, np.power(X,2), np.exp(X), np.cos(X), np.ones((len(data),1))], axis=1)

# Train the model with the expanded matrix and make the predictions
ld = np.array([0.1, 1, 10, 100, 200])

# Ridge regression implementation using cross-validation
# reg = RidgeCV(alphas = ld).fit(X_expanded, y)

# Lasso regression implementation including cross-validation
reg = LassoCV(alphas=ld,cv=10,selection='random',fit_intercept=False,random_state=42).fit(X_expanded, y)

# Obtaining the coefficients
coefs = reg.coef_
# print(reg.alpha_)
pd.DataFrame(coefs).to_csv("task1b/output.csv", header=None, index=None)