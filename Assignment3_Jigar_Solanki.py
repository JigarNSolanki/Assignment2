import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

# Provide the path of sonar.all-data file
f = open("/home/solanki/Desktop/Training/Assignment3/sonar.all-data",'r')
df = f.read()
df = np.array(df.split())

m = len(df)         # number of raws

X = [ [1] for i in range(m)]
Y = []

for i in range(m):
    dfi = np.array(df[i].split(sep=','))
    n = len(dfi) - 1                        # number of features
    Y.append(dfi[-1])
    for j in range(n):
        X[i].append(float(dfi[j]))
        
X = np.array(X)        
Y = np.array(Y)
#Y = Y.reshape((m,1))
#data = np.concatenate((X,Y),axis=1)
#df = pd.DataFrame(data)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.25, random_state=23)

##----------------Support Vector Classifier-----------------------------
modelSVM = svm.SVC(random_state=13, kernel="linear")
modelSVM.fit(Xtrain,Ytrain)
Ypred1 = modelSVM.predict(Xtest)
#print(confusion_matrix(Ytest,Ypred1))
print(accuracy_score(Ytest,Ypred1))

##------------- Decision Tree---------------------
#A=[]
#C=[]
modelDT = DecisionTreeClassifier(random_state=44, min_samples_split = 4, criterion="entropy")
modelDT.fit(Xtrain,Ytrain)
Ypred2 = modelDT.predict(Xtest)
#print(confusion_matrix(Ytest,Ypred2))
print(accuracy_score(Ytest,Ypred2))
#plt.plot(C,A)
#plt.xlabel("Minimum sample required")
#plt.ylabel("Accuracy")
#plt.savefig("Accuracy vs. Minimum sample requried to split...Decision tree.jpg")

##-----------Random Forest--------------------
modelRF = RandomForestClassifier(random_state=10, n_estimators=13, criterion="entropy")
modelRF.fit(Xtrain,Ytrain)
Ypred3 = modelRF.predict(Xtest)
#print(confusion_matrix(Ytest,Ypred3))
print(accuracy_score(Ytest,Ypred3))




