import numpy as np
import pandas as pd

# Importing the data
data = pd.read_csv('task0/test.csv')

# Isolating the numerical data 
dataNumerial = data.iloc[:, 1:np.shape(data)[1]]
rowID = data.iloc[:, 0]

# Calculating the mean of the 
y = dataNumerial.mean(1)

# Creating a dictionary with the relevant information and saving to a .csv file
output = pd.DataFrame({'Id': rowID, 'y': y})
output.to_csv("task0/output.csv", encoding='utf-8', index=False)