import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

dfw = pd.read_csv("winequality-white.csv", sep = ";")
dfr = pd.read_csv("winequality-red.csv", sep = ";")

#Independent variables
X = dfw[dfw.columns[0:11]]
#X = dfr[dfw.columns[0:11]]

#Array of dependent variable
Y = dfw.iloc[:,11:12].values.reshape(X.shape[0])
#Y = dfr.iloc[:,11:12].values.reshape(1599)


# Hypothesis function
def Yp(A,X,n):
    Ypred = np.ones((X.shape[0],1))
    A = A.reshape(1,n+1)
    for i in range(X.shape[0]):
        Ypred[i] = float(np.matmul(A, X[i]))
    Ypred = Ypred.reshape(X.shape[0])
    return Ypred

# Gradient function
def Grad(A, alpha, nl, Ypred, X, Y, n):
    m = len(Y)
    cost = np.ones(nl)
    for i in range(nl):
        A[0] = A[0] - (alpha/m)*sum(Ypred - Y)
        for j in range(1,n+1):
            A[j] = A[j] - (alpha/m)*sum((Ypred-Y) * X.transpose()[j])
        Ypred = Yp(A, X, n)
        cost[i] = (1/m) * 0.5 * sum(np.square(Ypred-Y))
    A = A.reshape(1,n+1)
    return A, cost    

alpha = 0.000001        # Learning Rate
nl = 100                # Number of Loop


# Linear Regression
n = X.shape[1]
X = np.concatenate((np.ones((X.shape[0],1)), X), axis = 1)
A = np.zeros(n+1)
Ypred = Yp(A, X, n)
A, cost = Grad(A,alpha,nl,Ypred,X,Y,n)

cost = list(cost)
ni = [i for i in range(1,nl+1)]
plt.plot(ni, cost)
plt.xlabel("Cost Function")
plt.ylabel("Number of iteration")


##-----------Final value of Y-------------------------
Yfinal = Yp(A,X,n)
for k in range(len(Yfinal)):
    Yfinal[k] = round(Yfinal[k])
print(Yfinal)


