import pandas as pd
import numpy as np

data = pd.read_csv('framingham.csv')

data.dropna(inplace=True)

# Separate the features (X) and the target variable (y)
X = data[['male' ,'age' ,'education' ,'currentSmoker' ,'cigsPerDay' ,'BPMeds' ,'prevalentStroke' ,
            'prevalentHyp' ,'diabetes' ,'totChol' ,'sysBP' ,'diaBP' ,'BMI' ,'heartRate' ,'glucose']]
y = data['TenYearCHD']

X = np.array(X)
y = np.array(y)

X = (X - X.mean(axis=0)) / X.std(axis=0)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (binary cross-entropy)
def cost(X, y, W, b):
    m=X.shape[0]
    cost=0.0
    for i in range(m):
        z_i = np.dot(X[i],W) + b
        f_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_i) - (1-y[i])*np.log(1-f_i)
    cost=cost/m 
    return cost

# Gradient descent for logistic regression
def gradient_descent(X, y, W, b, alpha, iterations):
    m=X.shape[0]
    for i in range(iterations):
        dW=np.zeros((n,))
        db=0
        for j in range(m):
            z_j = np.dot(X[j],W)+b
            err_j = sigmoid(z_j) - y[j]
            for k in range(n):
                dW[k]+= err_j*X[j,k]
            db+= err_j
        dW=dW/m
        db=db/m
        
        # Update weights
        W -= alpha * dW
        b -= alpha * db
    
    return W, b

m= X.shape[0]
n = X.shape[1]
W = np.zeros((n,))
b = 0

alpha = 0.05
iterations = 200

W, b = gradient_descent(X, y, W, b, alpha, iterations)

# Calculate accuracy

predictions = np.zeros((m,))
for i in range(m):
    z = np.dot(X[i],W) + b
    y_hat = sigmoid(z)
    if y_hat>=0.5:
        predictions[i]=1
    else:
        predictions[i]=0

correct=0
for i in range(m):
    if predictions[i]==y[i]:
        correct+=1

accuracy= correct*100/m

print('Accuracy=', accuracy, '%')

false_pos=0
false_neg=0
pos=0
neg=0
pred_neg=0
pred_pos=0
for i in range(m):
    if y[i]==1:
        pos+=1
        if predictions[i]==0:
            false_neg+=1
    else:
        neg+=1
        if predictions[i]==1:
            false_pos+=1

for i in range(m):
    if predictions[i]==0:
        pred_neg+=1
    else:
        pred_pos+=1
fal_pos = false_pos*100.0/neg
fal_neg = false_neg*100.0/pos

print('False positive=', fal_pos, '%')
print('False negative=', fal_neg, '%')