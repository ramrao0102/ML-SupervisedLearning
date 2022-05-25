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

random.seed(500)


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
    

#for i in range(5):
    #print(dataset[i])

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


# scaling the values in the dataset
# scaling is not used for modeling

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
# this function is used for modeling

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

#print(len(train1))

#print(len(test))

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

# train and test meanvals and stdevvals

# the below is just a check for mean and stdev by class_type

list1 =[]
list2 = []

for i in range(len(train)):
  
    if train[i][-1]==0:
       list1.append(train[i])
    else:
       list2.append(train[i])
           

# the above is just a check for mean and stdev by class_type    
       
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

# the below function summarizes:  stdeviation and mean by
# column in dataset

def summarize_dataset(dataset):
    
    summaries = []
    
    meanvals = mean(dataset)
    stdevvals = stdev(dataset)
   
    
    for i in range(len(dataset[0])-1):
        
        summaries.append((meanvals[i], stdevvals[i], len(dataset)))
    
    return(summaries)

#summaries = summarize_dataset(train)

#print(summaries)
        
# Split dataset by class values and return a dictionary

def separate_by_class(dataset):
    
    separated = dict()
    
    for i in range(len(dataset)):
        row = dataset[i]
        class_value = row[-1]
        
        if (class_value not in separated):
            separated[class_value] = list()
 
 # the below line assigns the row to the correct class value key in the dictionary       
 
        separated[class_value].append(row)
        
    return separated

separated = separate_by_class(train)

#print(separated)
            
# Split dataset and calculate statistics

def summarize_by_class(dataset):
    
    separated = separate_by_class(dataset)
    summaries = dict()
    
    for class_value, rows in separated.items():
        #print(len(rows))
        summaries[class_value] = summarize_dataset(rows)
        
    return summaries

summaries = summarize_by_class(train)

#print(summaries)

# Calculate the Gaussian Probability Density Function for x

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2/(2*stdev**2)))
    
    return (1/(sqrt(2*pi)*stdev))* exponent   

# this below method calculates the probabilities for each class
# for the dataset

def calculate_class_probabilities(summaries, row):
    
    total_rows = sum([summaries[label][0][2] for label in summaries] )
    probabilities = dict()
    
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
        
    return probabilities


predictions = []

# the 2 lines of code below  places the probability for each class label
# for the dataset

for i in range(len(train)):
    predictions.append(calculate_class_probabilities(summaries, train[i]))     

#print(predictions)

# the best label are the max of the two labels are calculated here.
# the below is to estimate the argmax of the y belonging to Y per
# Naive Bayes Method

best_labels =[]

for i in range(len(predictions)):
    best_label = None
    best_prob = -1
    for class_value, probability in predictions[i].items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
            
    best_labels.append(best_label)

actual = []

# Evaluate algorithm for the train dataset

print("Naive_Bayes_Train_Dataset")
        
for row in train:
    actual.append(row[-1])


conf_matrix = confusion_matrix(actual, best_labels)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual, best_labels)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual,best_labels)

print(rmse)

accuracy = accuracy_metric(actual, best_labels)

print(accuracy)


# Below is Evaluation and Performance Metrics for Validation Dataset

print("Naive_Bayes_Val_Dataset")

valpredictions = []

for i in range(len(val)):
    valpredictions.append(calculate_class_probabilities(summaries, val[i]))
    

best_labels =[]

for i in range(len(valpredictions)):
    best_label = None
    best_prob = -1
    for class_value, probability in valpredictions[i].items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
            
    best_labels.append(best_label)

actual = []

# Evaluate the algorithm for the test dataset
        
for row in val:
    actual.append(row[-1])

conf_matrix = confusion_matrix(actual, best_labels)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual, best_labels)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual,best_labels)

print(rmse)

accuracy = accuracy_metric(actual, best_labels)

print(accuracy)

# Below is Evaluation and Performance Metrics for Test Dataset

print("Naive_Bayes_Test_Dataset")

testpredictions = []

for i in range(len(test)):
    testpredictions.append(calculate_class_probabilities(summaries, test[i]))
    

best_labels =[]

for i in range(len(testpredictions)):
    best_label = None
    best_prob = -1
    for class_value, probability in testpredictions[i].items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
            
    best_labels.append(best_label)

actual = []

# Evaluate the algorithm for the test dataset
        
for row in test:
    actual.append(row[-1])

conf_matrix = confusion_matrix(actual, best_labels)

print(conf_matrix)

precision, recall, F1_score = precision_recall_f1(actual, best_labels)

print(precision, recall, F1_score)  

rmse = root_mean_square_error(actual,best_labels)

print(rmse)

accuracy = accuracy_metric(actual, best_labels)

print(accuracy)
