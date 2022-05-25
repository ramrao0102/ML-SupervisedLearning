# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 21:45:40 2022

@author: ramra
"""

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
from math import pi
from math import sqrt
from random import random

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


#for i in range(5):
    #print(standardized_dataset[i])   

for i in range(len(standardized_dataset)):
    if (standardized_dataset[i][len(dataset[0])-1]) == 'Kecimen':
       standardized_dataset[i][len(dataset[0])-1]  = 1
        
    else:
        standardized_dataset[i][len(dataset[0])-1]  = 0
    

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

# the below is just a check for mean and stdev by class_type

list1 =[]
list2 = []

for i in range(len(train)):
  
    if train[i][-1]==0:
       list1.append(train[i])
    else:
       list2.append(train[i])
       
print(mean(list1)) 

print(mean(list2))      


# the above is just a check for mean and stdev by class_type    
       
meantrainvals = mean(train)

print(meantrainvals)

stdevtrainvals = stdev(train)

meantestvals = mean(test)

stdevtestvals = stdev(test)

# Calculate the mean, stdev, and count for each column in dataset

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

# Below Functions are to Generate TPR and FPR Plots


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

def AUC(proba, actual1):

    # Area Under Curve Determinations

    # for Model 1, Validation Dataset


    Youden_Index, final_index, TPR, FPR = TPRFPR(proba, actual1)
    
    AUC1 = 0.0

    for i in range(len(FPR)):

        if i == 0:
        
            prev_coordinate = FPR[i]
            AUC1 =0.0

        if i >0:
    
            AUC1 += (1/2)* (TPR[i] +TPR[i])*(-FPR[i] + prev_coordinate)   


        prev_coordinate = FPR[i]

    print("AUC from Fuction Built from Raw FPR and TPR Data, Model 1 Validation Dataset")
    
    print(AUC1)
 
    return 0

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

# this is a maximum of 2 hidden layer implementation

def initialize_neural_network(n_inputs, n_hidden, n_hidden2, n_outputs):
    
    hidden1_layer = list()
    hidden2_layer = list()
    output1_layer = list()
    network = list()    
    
    if n_hidden != 0:
    
        for i in range(n_hidden):
            hidden_dict = {'weights': None}
            hidden_layer = []
            for j in range(n_inputs +1):
                hidden_layer.append(random())        
            
            hidden_dict['weights'] = hidden_layer
        
            hidden1_layer.append(hidden_dict)
    
        for i in range(n_hidden2):
            hidden2_dict = {'weights': None}
            hidden22_layer = []
            for j in range(n_hidden + 1):
                hidden22_layer.append(random())
        
            hidden2_dict['weights'] = hidden22_layer
        
            hidden2_layer.append(hidden2_dict)
    
    else:
        
        for i in range(n_hidden2):
            hidden2_dict = {'weights': None}
            hidden22_layer = []
            for j in range(n_inputs +1):
                hidden22_layer.append(random())
        
            hidden2_dict['weights'] = hidden22_layer
        
            hidden2_layer.append(hidden2_dict)
        
    
    for i in range(n_outputs):
        output_dict = {'weights': None}
        output_layer = []
        for j in range(n_hidden2 + 1):
            output_layer.append(random())
            
        output_dict['weights'] = output_layer
            
        output1_layer.append(output_dict)
    
    if n_hidden !=0:
        network.append(hidden1_layer)
        network.append(hidden2_layer)
        network.append(output1_layer)
        
    if n_hidden == 0:
        network.append(hidden2_layer)
        network.append(output1_layer)
    
    return network
        
# Forward Propogation

def activate(weights, inputs):
    
    activation = weights[-1]
    #print("are we here")
    #print(weights)
    for i in range(len(weights)-1):
        
        activation += weights[i] * inputs[i]
        
    return activation

# Sigmoid Activation Filter

def transfer(activation):
    
    return 1.0/(1.0+exp(-activation))


def forward_propogate(network, row):
    inputs = row
    
    for layer in network:
        new_inputs = []
        
        for neuron in layer:
                      
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
             
            new_inputs.append(neuron['output'])
            
        inputs = new_inputs
   
    return inputs
    
 
# Backpropogate error and associated calculations


def transfer_derivative(output):
    
    return output *(1.0 - output)


def backward_propogate_error(network, expected):
    
    for i in reversed(range(len(network))):
    
        layer = network[i]
        errors = list()

# if is to check if i is output layer or not
# errors are calcuated differently for output and hidden layers

        if i != len(network) -1:
            
            for j in range(len(layer)):
                
                error = 0.0
                
                for neuron in network[i+1]:
                
                    error += (neuron['weights'][j]*neuron['delta'])
    
                errors.append(error)
                
        else:
            
            for j in range(len(layer)):
                
                neuron = layer[j]
                errors.append(expected[j]-neuron['output'])
                
        for j in range(len(layer)):
            
            neuron = layer[j]
            
            neuron['delta'] = errors[j]*transfer_derivative(neuron['output'])
    
# Train Network

# We need to train the network using the stochastic gradient descent


def update_weights(network, row, l_rate):
    
    for i in range(len(network)):
        inputs = row[:-1]

    if i != 0:
        inputs = [neuron['output'] for neuron in network[i-1]]
    
    for neuron in network[i]:
        for j in range(len(inputs)):
            neuron['weights'][j] += l_rate *neuron['delta']*inputs[j]
        neuron['weights'][-1] += l_rate *neuron['delta']
        
# Train the network        
        

def train_network(network, train, l_rate, n_epoch, n_outputs):
    Error = []
    for epoch in range(n_epoch):
    
        sum_error =0 
        for row in train:
            outputs = forward_propogate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]- outputs[i])**2 for i in range(len(expected))])
            backward_propogate_error(network, expected)
            update_weights(network, row,l_rate)
        
        print('>epoch=%d, lrate=%.3f, error = %.3f' %(epoch, l_rate, (sum_error/len(train))**0.5))
        Error.append((sum_error/len(train))**0.5)

# the error is reflective of the RMSE at the end of each iteration of a training episode
        
    return Error

def predict(network, row):
    outputs = forward_propogate(network, row)
    #print(outputs, outputs.index(max(outputs)))
    return outputs, outputs.index(max(outputs))

n_inputs = len(train[0])-1

n_outputs = len(set(row[-1] for row in train)) 

network =   initialize_neural_network(n_inputs, 0, 20, n_outputs)

Error = train_network(network, train, 0.5, 2000, n_outputs)

for layer in network:
    print(layer)
    
actual =[]
predictions = [] 
    
for row in train:
    outputs, prediction = predict(network, row)
    #print('Expected=%d, Got =%d' % (row[-1], prediction))
    actual.append(row[-1])
    predictions.append(prediction)
    
# Check Performance Metrics on Train Dataset    

print("ANN_Train_Dataset")


conf_matrix = confusion_matrix(actual, predictions)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual, predictions)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual,predictions)

print(rmse)   

accuracy = accuracy_metric(actual, predictions)

print(accuracy)

# Evaluate and Check Performance Metrics on Validation Dataset

print("ANN_Val_Dataset")

actual_val =[]
predictions_val = [] 
    
for row in val:
    outputs, prediction = predict(network, row)
    #print('Expected=%d, Got =%d' % (row[-1], prediction))
    actual_val.append(row[-1])
    predictions_val.append(prediction)
    
# Check Performance Metrics on Validation Dataset    

conf_matrix = confusion_matrix(actual_val, predictions_val)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual_val, predictions_val)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual_val, predictions_val)

print(rmse)   

accuracy = accuracy_metric(actual_val, predictions_val)

print(accuracy)

# Now check on Test Dataset

actual1 =[]

predictions1 = []

test_proba = []
 
for row in test:
    
    #print(j)
    outputs, prediction1 = predict(network, row)
      #print('Expected=%d, Got =%d' % (row[-1], prediction))
    actual1.append(row[-1])
    predictions1.append(prediction1)
    test_proba.append(outputs)

print(len(predictions1)) 
print(len(actual1))        
#print(len(test_proba))

# FPR, TPR, and AUC Calculations for Test_Dataset

print("ANN_Test_Dataset")

predictions_test = probability_threshold(test_proba)

RX, RY = random_predictor()

Youden_Index, final_index, TPR, FPR = TPRFPR(test_proba, actual1)

AUC = AUC(test_proba, actual1)

print("Probability Threshold with Highest Index", final_index)        
plt.figure(figsize=(3, 3))
plt.plot(FPR, TPR)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["ANN 2HL:10N, Test Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for Model 1 Validation Dataset is:")
print(Youden_Index)

# Performance Metrics for Neural Networks Test Dataset

conf_matrix = confusion_matrix(actual1, predictions1)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual1, predictions1)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual1,predictions1)

print(rmse)    

accuracy = accuracy_metric(actual1, predictions1)

print(accuracy)  

plt.figure(figsize=(3, 3))
plt.plot(Error)
plt.xlabel("Number of Iterations.", size = 8,)
plt.ylabel("Error During Training", size = 8)
plt.legend(["ANN 2HL:10N Training"], loc ="upper right", prop = {'size': 8})
plt.show()           