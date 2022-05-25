
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 01:07:44 2022

@author: ramra
"""

# This is Ramkishore Rao's supervised learning  project
# This includes the code for Stochastic Gradient Descent and KNN

# First create a function to load the csv file

import csv
from csv import reader 
import random
from random import seed
from random import randrange
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from math import exp

seed(500)

def load_csv(filename):
    
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    
    return dataset

#dataset is the data that will be manipulated by scaling and/or standardizing

dataset = load_csv("C:/Data Science and Analytics/CS 5033/Supervised Learning Project/Raisin_Dataset/Raisin_Dataset/Raisin_Dataset.csv")

# data is the dataset that will be kept as original dataset without scaling and/or standardizing 

data = load_csv("C:/Data Science and Analytics/CS 5033/Supervised Learning Project/Raisin_Dataset/Raisin_Dataset/Raisin_Dataset.csv")

# Remove first row from dataset which are the variable names

del dataset[0]
del data[0]

# Convert String to Float

def StringtoFloat(dataset, column):
    for row in dataset:
        if row !=0:
            row[column] = float(row[column].strip())
        
for i in range(7):
    StringtoFloat(dataset, i)

# this is just a check for first 5 rows of the dataset
# and the type of the variables in the column of the dataset
    
for i in range(5):
    print(dataset[i])

dataset1= dataset

#for i in range(5):
    #for j in range(len(dataset[0])):
        #print(type(dataset[i][j]))
        
# Function to Scale the Float Values of Columns in Dataset

# function to find minimum and maximum in column values in dataset

def minmax(dataset):
    
    minmax = list()

    for i in range(len(dataset[0])-1):
        value_min =500000
        value_max=0
        for row in range(len(dataset)):
            col_values = dataset[row][i]
            if value_min > col_values:
                value_min = col_values
            if value_max < col_values:
                value_max = col_values
        minmax.append([value_max, value_min])
    return minmax
  

#scaled_dataset = dataset
# scaling approach is not used for modeling

# scaling the values in the dataset

#def scale_dataset(dataset):
    
    #minmaxvalues = minmax(dataset)
    #for i in range(len(dataset[0])-1):
        #for row in range(len(dataset)):
            #scaled_dataset[row][i] = (dataset[row][i] - minmaxvalues[i][1])/(minmaxvalues[i][0] -minmaxvalues[i][1] )
    
    #return scaled_dataset


# this is a test code just to check scaled_dataset

#for i in range(5):
    #print(scaled_dataset[i])   

# normalizing the dataset  and calculating standardized value

# function to find mean of col_values in dataset

def mean(dataset):
    
    mean = list()
    
    for i in range(len(dataset[0])-1):
        sum =0
        for row in range(len(dataset)):
            sum += dataset[row][i]
        print(sum)    
        ave = sum/(len(dataset))
        mean.append(ave)

    return mean


# function to find standard deviation of col_values in dataset

def stdev(dataset):

    stdev = list()
    meanvals = mean(dataset)

    for i in range(len(dataset[0])-1):
        vals = 0
        for row in range(len(dataset)):
            vals += ((dataset[row][i]- meanvals[i])**2)
    
        stdevcol = (vals/(len(dataset)-1))**0.5
        stdev.append(stdevcol)
    
    return stdev
       

# finally, a function for standardized col_values
# this fuction is used for modeling; we are using standardized dataset

def standardized_value(dataset):
    
    standardized_value = dataset
    
    meanvals = mean(dataset)
    stddevvals = stdev(dataset)
    
    for i in range(len(dataset[0])-1):
        for row in range(len(dataset)):
            standardized_value[row][i] = ((dataset[row][i]- meanvals[i])/stddevvals[i])
            
    return standardized_value

standardized_dataset = standardized_value(dataset)


for i in range(5):
    print(standardized_dataset[i])   
    
# Function to split the dataset into train and test components

def train_test_split(data, test_split):
    
    train1 = list()
    
    train_size = (1-test_split)*len(data)
    test = list(data)
    
    while len(train1) < train_size:
        index = randrange(len(test))
        train1.append(test.pop(index))
    
    return train1, test

train1, test = train_test_split(standardized_dataset, 0.2)

print(len(train1))

print(len(test))

# Function for hold-out validation and train split from train dataset

def validation_split(data, val_split):
    
    train = list()
    
    train_size = (1-val_split)*len(data)
    val = list(data)
    
    while len(train) < train_size:
        index = randrange(len(val))
        train.append(val.pop(index))
    
    return train, val

train, val = validation_split(train1, 0.15)


def accuracy_metric(actual, predicted):
    
    correct = 0
    
    for i in range(len(actual)):
        
        if actual[i] == predicted[i]:
            
            correct += 1
            
    accuracy = correct/float(len(actual)) 
    
    return accuracy
            
# quick test to be replaced by real values later on

#actual = [0,0,1,0,1]

#predicted = [1,1,1,0,1]

#accuracy = accuracy_metric(actual, predicted)

#print(accuracy/100)

# Second Function is for the development of the confusion matrix

def confusion_matrix(actual, predicted):
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative =0

# initialize a confusion_matrix
    
    confusion_matrix = np.zeros((2,2))
    
    for i  in range(len(actual)):
        
        if  predicted[i] == 1:
            if actual[i] == 1:
                true_positive += 1
                
            if actual[i] == 0:
                false_positive += 1 
                
        if predicted[i] == 0:
            if actual[i] == 0:
                true_negative += 1
            
            
            if actual[i] == 1:
                false_negative += 1
                
    confusion_matrix[1][1] = true_positive
    
    confusion_matrix[1][0] = false_negative
    
    confusion_matrix[0][1] = false_positive
    
    confusion_matrix[0][0] = true_negative
    
    return confusion_matrix

# the below is a simple test to check accuracy of confusion matrix

#actual = [0,0,1,1,1]

#predicted = [1,0,1,0,1]
               
#conf_matrix = confusion_matrix(actual, predicted)        

#print(conf_matrix)

# Third Function to Estimate Root Mean Square Error

def root_mean_square_error(actual, predicted):
    
    sum = 0
    
    for i  in range(len(actual)):
        
        predict_error = predicted[i] - actual[i] 
        sum += predict_error**2
        
    rmse = sum/float(len(actual))
    
    return (rmse)**0.5

# quick check for RMSE calculation

#actual = [0,0,1,1,1]

#predicted = [1,0,1,0,1]
        
#rmse = root_mean_square_error(actual, predicted)

#print(rmse)        

def precision_recall_f1(actual, predicted):
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative =0

    for i  in range(len(actual)):
    
        if  predicted[i] == 1:
            if actual[i] == 1:
                true_positive += 1
            
            if actual[i] == 0:
                false_positive += 1 
            
        if predicted[i] == 0:
            if actual[i] == 0:
                true_negative += 1
        
        
            if actual[i] == 1:
                false_negative += 1
    
    precision = true_positive/(true_positive + false_positive)
    
    recall = true_positive/(true_positive + false_negative)
    
    F1_score = 2 *(precision *recall)/(precision + recall)
    
    return precision, recall, F1_score

#quick check for precision, recall, and F1_score calculations

#actual =    [0,0,1,1,1,0, 0]

#predicted = [1,0,1,0,1,0, 1]

#conf_matrix = confusion_matrix(actual, predicted)        

#print(conf_matrix)

#precision, recall, F1_score = precision_recall_f1(actual, predicted)

#print(precision, recall, F1_score))

def probability_threshold(proba):

    predictions = []

    for i in range(0,1001,1):
        k = float(i/1000)
        label3 = []
        for j in range(len(proba)):
        
            if proba[j][1] > k:
                label3.append(1)
            else:
                label3.append(0)

        predictions.append(label3)

    return predictions

# Random Predictor

def random_predictor():

    RX = []
    RY = []

    for i in range(0,1001, 1):
        k = float(i/1000)
        
        RX.append(k)
        RY.append(k)
        
    return RX, RY

def TPRFPR(proba, actual1):

    RX, RY = random_predictor()
    
    predictions = probability_threshold(proba)

    TPR = []
    FPR = []
    Youden_Index = 0

    for i in range(0,1001, 1):
        k = float(i/1000)
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for j in range(len(predictions[i])):
            if predictions[i][j] ==0:
                if actual1[j] == 0:
                    tn += 1
                else:
                    fn += 1
        
            if predictions[i][j] == 1:
                if actual1[j]  == 1:
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
    
        if (true_positive_rate1 - false_positive_rate1) > Youden_Index:
            Youden_Index = (true_positive_rate1 - false_positive_rate1)
            final_index = k
  
    
        TPR.append(true_positive_rate1)
        FPR.append(false_positive_rate1)
     
    return Youden_Index, final_index, TPR, FPR

def AUC(proba, actual):

    # Area Under Curve Determinations

    # for Model 1, Validation Dataset


    Youden_Index, final_index, TPR, FPR = TPRFPR(proba, actual)
    
    AUC1 = 0.0

    for i in range(len(FPR)):

        if i == 0:
        
            prev_coordinate = FPR[i]
            AUC1 =0.0

        if i >0:
    
            AUC1 += (1/2)* (TPR[i] +TPR[i])*(-FPR[i] + prev_coordinate)   


        prev_coordinate = FPR[i]

    print("AUC from Fuction Built from Raw FPR and TPR Data, Test Dataset")
    
    print(AUC1)
 
    return 0


# first convert the target variable to 0 and 1 binary class
# Kecimen is 1 and Besni is 0


for i in range(len(standardized_dataset)):
    if (standardized_dataset[i][len(dataset[0])-1]) == 'Kecimen':
       standardized_dataset[i][len(dataset[0])-1]  = 1
        
    else:
        standardized_dataset[i][len(dataset[0])-1]  = 0
        
# Function for Logistic Regression with Stochastic Gradient Descent

#for i in range(len(standardized_dataset)):
    #print(standardized_dataset[i])

def predict(row, coefficients):
	sigmas = coefficients[0]
	for i in range(len(row)-1):
		sigmas += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-sigmas))


def coefficients_sgd(train, l_rate, n_epoch):
    Error = []
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
             
            sigmas = predict(row, coef)
            error = (row[-1] - sigmas)  
            sum_error += (error)**2
            coef[0] = coef[0] +l_rate*error*sigmas*(1.0-sigmas)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * sigmas * (1.0 - sigmas) * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, (abs(2*sum_error)/len(train))**0.5))
        Error.append(((abs(2*sum_error)/len(train))**0.5))

    return coef, Error


# Function to estimate predictions for train dataset

def logistic_regression_train(train, l_rate, n_epoch):
	predictions1 = list()
	coef, Error = coefficients_sgd(train, l_rate, n_epoch)
	for row in train:
		y1hat = predict(row, coef)
		y1hat = round(y1hat)
        
		predictions1.append(y1hat)
	return coef, predictions1, Error
    
l_rate = 0.1
n_epoch = 1000

coef, predictions1, Error = logistic_regression_train(train, l_rate, n_epoch)

actual1 = []

# Evaluation for Train Dataset

print("Evaluation_SGD_Train_Dataset")

for row in train:
    actual1.append(row[-1])

conf_matrix1 = confusion_matrix(actual1, predictions1)

print(conf_matrix1)

precision, recall, F1_score = precision_recall_f1(actual1, predictions1)

print(precision, recall, F1_score)

rmse = root_mean_square_error(actual1, predictions1)

print(rmse)

accuracy = accuracy_metric(actual1, predictions1)

print(accuracy)

# Error During Training
plt.figure(figsize=(3, 3)) 
plt.plot(Error)
plt.xlabel("Number of Iterations.", size = 8,)
plt.ylabel("Error During Training", size = 8)
plt.legend(["SGD Training"], loc ="upper right", prop = {'size': 8})
plt.show()  

# Function to estimate predictons for val dataset

def logistic_regression_val(val):
    predictions_val = list()    
    val_proba = list()
   
    for row in val:
        yhat = predict(row, coef)
        yhatneg = 1 - yhat
        yhat_round = round(yhat)
        
        predictions_val.append(yhat_round)
        val_proba.append([yhatneg, yhat])
    return predictions_val, val_proba

predictions_val, val_proba = logistic_regression_val(val)

actual_val =[]

# Evaluation for Val dataset

print("Evaluation_SGD_Val_Dataset")

for row in val:
    actual_val.append(row[-1])  

# Performance Metrics for Val Dataset

conf_matrix = confusion_matrix(actual_val, predictions_val)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual_val, predictions_val)

print(precision, recall, F1_score)

rmse = root_mean_square_error(actual_val,predictions_val)

print(rmse)

accuracy = accuracy_metric(actual_val, predictions_val)

print(accuracy)

# Function to estimate predictions for test dataset

def logistic_regression_test(test):
    predictions = list()    
    test_proba = list()
   
    for row in test:
        yhat = predict(row, coef)
        yhatneg = 1 - yhat
        yhat_round = round(yhat)
        
        predictions.append(yhat_round)
        test_proba.append([yhatneg, yhat])
    return predictions, test_proba

predictions, test_proba = logistic_regression_test(test)

actual =[]

# Evaluation for test dataset

print("Evaluation_SGD_Test_Dataset")

for row in test:
    actual.append(row[-1])

#Calculation for FPR, TPR, AUC Test Dataset

predictions_test = probability_threshold(test_proba)

RX, RY = random_predictor()

Youden_Index, final_index, TPR, FPR = TPRFPR(test_proba, actual)

AUC = AUC(test_proba, actual)

print("Probability Threshold with Highest Index", final_index)  
plt.figure(figsize=(3, 3))      
plt.plot(FPR, TPR)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["SGD Test Dataset"], loc ="lower right", prop = {'size': 8})

plt.show()   

print("Highest Youden_Index for Test Dataset is:")
print(Youden_Index)

# Performance Metrics for Test Dataset

conf_matrix = confusion_matrix(actual, predictions)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual, predictions)

print(precision, recall, F1_score)


rmse = root_mean_square_error(actual,predictions)

print(rmse)

accuracy = accuracy_metric(actual, predictions)

print(accuracy)

# Function for K Nearest Neighbors Algorithm

def euclidean_distance (row1, row2):

    distance = 0.0

    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    
    return distance**0.5


def get_neighbors(train, test_row, num_neighbors):
    
    distances = list()
    
    for row in train:
        dist = euclidean_distance(test_row, row)
        distances.append((row, dist))
        
    distances.sort(key = lambda tup: tup[1])

    neighbors = list()
    
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors

def pred(train, test_row, num_neighbors):
    
    neighbors = get_neighbors(train, test_row, num_neighbors)
    
    output_value = list()
    
    for row in neighbors:
        output_value.append(row[-1])
    
    prediction = sum(output_value)/len(output_value)
    
    return prediction

# Function to predict KNN for train dataset

def KNN_train(train, num_neighbors):
    predictions1 = list()
    
    for row in train:
        output = pred(train, row, num_neighbors)
        output = round(output)
        
        predictions1.append(output)
    
    return predictions1

num_neighbors = 4

predictions1 = KNN_train(train, num_neighbors)

actual1 = []

print("Evaluation_KNN_Train_Dataset")

for row in train:
    actual1.append(row[-1])

conf_matrix1 = confusion_matrix(actual1, predictions1)

print(conf_matrix1)

precision, recall, F1_score = precision_recall_f1(actual1, predictions1)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual1,predictions1)

print(rmse)

accuracy = accuracy_metric(actual1, predictions1)

print(accuracy)

# Function to predict KNN for val dataset

print("Evaluation_KNN_Val_Dataset")

def KNN_val(train, val, num_neighbors):
    predictions = list()
    
    for row in val:
        output = pred(train, row, num_neighbors)
        output = round(output)
        
        predictions.append(output)
  
    return predictions

num_neighbors = 4

predictions = KNN_val(train, val, num_neighbors)

print (predictions)
    
actual = []

for row in val:
    actual.append(row[-1])

conf_matrix = confusion_matrix(actual, predictions)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual, predictions)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual,predictions)

print(rmse)

accuracy = accuracy_metric(actual, predictions)

print(accuracy)

# Function to predict KNN for test dataset

print("Evaluation_KNN_Test_Dataset")

def KNN_test(train, test, num_neighbors):
    predictions = list()
    
    for row in test:
        output = pred(train, row, num_neighbors)
        output = round(output)
        
        predictions.append(output)
  
    return predictions

num_neighbors = 4

predictions = KNN_test(train, test, num_neighbors)

print (predictions)
    
actual = []

for row in test:
    actual.append(row[-1])

conf_matrix = confusion_matrix(actual, predictions)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual, predictions)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual,predictions)

print(rmse)

accuracy = accuracy_metric(actual, predictions)

print(accuracy)

# Single Perceptron

l_rate = 0.01
n_epoch = 100

def predict (row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i+1]*row[i]
        
    if activation >=0.0:
        return 1
    else:
        return 0
    
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]

    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2.0
            weights[0] = weights[0] + l_rate*error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] +l_rate*error*row[i]
        #print('epocch = %d, lrate = %.3f, error = %.3f' %(epoch, l_rate, (sum_error/len(row))**0.5))

    return weights

# Perceptron with Stochastic Gradient Descent

def perceptron(train, test, l_rate, n_epoch):

    predictions = list()

    weights = train_weights(train, l_rate, n_epoch)  
    
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
        
    return weights, predictions

weights, predictions = perceptron(train, test, l_rate, n_epoch)

#print(weights)

print("Perceptron Predictions")

print(predictions)

actual = []

# Evaluation for Test Dataset

for row in test:
    actual.append(row[-1])

conf_matrix = confusion_matrix(actual, predictions)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual, predictions)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual,predictions)

print(rmse)            

accuracy = accuracy_metric(actual, predictions)

print(accuracy)