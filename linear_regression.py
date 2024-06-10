import pandas as pd
import numpy as np
import math

data = pd.read_csv('LifeExpectancyDataset.csv')

#drop rows with empty cells
data = data.dropna()

X = data[['Year', 'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI', 
          'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years', 
          'thinness 5-9 years', 'Income composition of resources', 'Schooling']]
y = data['Life expectancy']

X = np.array(X)
y = np.array(y)

X = (X - X.mean(axis=0)) / X.std(axis=0)

#print(X.shape)
m=X.shape[0]
ones=np.ones((m,1))
X = np.hstack((ones, X))
#print(X.shape)

theta = np.zeros(X.shape[1])

# Define hyperparameters
alpha = 0.15
num_iterations = 2000

for i in range(num_iterations):
    gradients = (1/m)*X.T @ (X @ theta - y) 
    theta = theta - alpha*gradients

predictions = X @ theta
error = predictions - y
mse= (1/m) * np.dot(error,error)
rmse = math.sqrt(mse)
print('RMSE Loss=', rmse)