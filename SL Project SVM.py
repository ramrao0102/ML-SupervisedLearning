# -*- coding: utf-8 -*-
"""
Created on Tue May  3 19:35:18 2022

@author: ramra
"""

# This partocular module uses all SKLearn's In Built Methods
# Except the Function to Calculate the ROC Curve

# SVM Classifier

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Data Science and Analytics/CS 5033/Supervised Learning Project/Raisin_Dataset/Raisin_Dataset/Raisin_Dataset.csv'
                )

print(df.head())

for i in df.index:
    if df['Class'][i] == 'Kecimen':
        df['Class'][i] = 1
    else:
        df['Class'][i] = 0

print(df.head())

print(df.dtypes)

X = df.iloc[:, :-1]

Y = df.iloc[:,-1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2)

print(X_train.head(), len(X_train))

print(X_test.head(), len(X_test))

print(y_train.head(), len(y_train))

print(y_test.head(), len(y_test))

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, random_state = 1, test_size = 0.15)

print(X_train1.head(), len(X_train1))

print(X_val1.head(), len(X_val1))

print(y_train1.head(), len(y_train1))

print(y_val1.head(), len(y_val1))

print(type(y_train1))

y_train11 = y_train1.to_numpy()

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
                 
predictions = clf.predict(X_train1)  

probability = clf.predict_proba(X_train1)  

# Peformance Metrics for Train Dataset

print("SVM_Linear_Train_Dataset")

accuracy = accuracy_score(y_train11, predictions)
precision = precision_score(y_train11, predictions)
recall = recall_score(y_train11, predictions)
F1_score = f1_score(y_train11, predictions)
confusion_mat_train = confusion_matrix(y_train11, predictions)
AUC_score = roc_auc_score(y_train11,probability[:,1])
mse = mean_squared_error(y_train11, predictions, squared=False)

print(accuracy, precision, recall, F1_score)

print(confusion_mat_train)
print(AUC_score)

print(mse)

#for i in range(len(predictions)):
    #print( probability[i], predictions[i], y_train11[i])

# for validation dataset

print("SVM_Linear_Val_Dataset")

predictions1 = clf.predict(X_val1)  

probability1 = clf.predict_proba(X_val1)  

y_val11 = y_val1.to_numpy()

# Peformance Metrics

accuracy1 = accuracy_score(y_val11, predictions1)
precision1 = precision_score(y_val11, predictions1)
recall1 = recall_score(y_val11, predictions1)
F1_score = f1_score(y_val11, predictions1)
confusion_mat_val1 = confusion_matrix(y_val11, predictions1)
AUC_score1 = roc_auc_score(y_val11, probability1[:,1])
mse1 = mean_squared_error(y_val11, predictions1, squared=False)

print(accuracy1, precision1, recall1, F1_score)

print(confusion_mat_val1)

print(AUC_score1)
print(mse1)
    
#for test dataset

predictions2 = clf.predict(X_test)  

probability2 = clf.predict_proba(X_test)  

y_test11 = y_test.to_numpy()

# Peformance Metrics

print("SVM_Linear_Test_Dataset")

accuracy2 = accuracy_score(y_test11, predictions2)
precision2 = precision_score(y_test11, predictions2)
recall2 = recall_score(y_test11, predictions2)
F1_score = f1_score(y_test11, predictions2)
confusion_mat_test2 = confusion_matrix(y_test11, predictions2)
AUC_score2 = roc_auc_score(y_test11, probability2[:,1])
mse2 = mean_squared_error(y_test11, predictions2, squared=False)

print(accuracy2, precision2, recall2, F1_score)

print(confusion_mat_test2)

print(AUC_score2)

print(mse2)

# ROC Curve Display Using Function Developed on Project

predictions11 = []

for i in range(0,1001,1):
    k = float(i/1000)
    label1 = []
    for j in range(len(probability2)):
        
        if probability2[j][1] > k:
            label1.append(1)
        else:
            label1.append(0)

    predictions11.append(label1)
  
# Random Predictor

RX = []
RY = []


for i in range(0,1001, 1):
    k = float(i/1000)
        
    RX.append(k)
    RY.append(k)
    
# Test Dataset ROC Curve Function

TPR1 = []
FPR1 = []
Youden_Index1 = 0

for i in range(0,1001, 1):
    k = float(i/1000)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for j in range(len(predictions11[i])):
        if predictions11[i][j] ==0:
            if y_test11[j] == 0:
                tn += 1
            else:
                fn += 1
        
        if predictions11[i][j] == 1:
            if y_test11[j]  == 1:
                tp += 1
            else:
                fp += 1
        
    if (tp+fn) != 0:
        true_positive_rate1 = tp/(tp+fn)
    else:
        true_positive_rate1 =0
    
    
    
    if (tn +fp) !=0:
        false_positive_rate1 = fp/(tn+fp)
    else:
        false_positive_rate1 = 0
    
    if (true_positive_rate1 - false_positive_rate1) > Youden_Index1:
        Youden_Index1 = (true_positive_rate1 - false_positive_rate1)
        final_index = k
       
    
    
    TPR1.append(true_positive_rate1)
    FPR1.append(false_positive_rate1)

print("Probability Threshold with Highest Index", final_index)   
plt.figure(figsize=(3, 3))     
plt.plot(FPR1, TPR1)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["SVM Model for Test Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for SVM Model for Test Dataset is:")
print(Youden_Index1)
    
    

               