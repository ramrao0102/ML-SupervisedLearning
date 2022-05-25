# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 20:10:38 2022

@author: ramra
"""

# Exploratory Data Analytics

import csv
from csv import reader 
import random
from random import seed
from random import randrange
import numpy as np
import math
import matplotlib.pyplot as plt
from math import exp
from math import pi
from math import sqrt
from random import random
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly.tools as tls

seed(1)


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


# scaling the values in the dataset; not using scaled data for this exercise

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

meanvals = mean(dataset)

stdevvals = stdev(dataset)

for i  in range(len(meanvals)):
    print ('%.2f'%meanvals[i])

for i  in range(len(meanvals)):
    print ('%.2f'%stdevvals[i])     


# finally, a function for standardized col_values

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
          
nparray = np.asarray(standardized_dataset) 

transposed = np.transpose(nparray)

print(transposed)

# creating 2 Numpy Arrays Based on Values of Label

# finding array index positions with label ==0

zeros = np.where(transposed[7]==0)[0]

print(zeros[0])

# Because the arrays can be just sliced into two to create two sub-arrays
# that contain code for slicing full arrays into two sub-arrays by Class type.

sub_arrays = np.split(transposed, 2, axis=1)

transposed_label1 = sub_arrays[0]

transposed_label2 = sub_arrays[1]


print(transposed_label1[0:1,:])

# the below code is only used to do scatter matrix plots and andrwws_curves in plotly   
        
df = pd.read_csv('C:/Data Science and Analytics/CS 5033/Supervised Learning Project/Raisin_Dataset/Raisin_Dataset/Raisin_Dataset.csv'
                )
print(df.head())

# this is an adrews curve plot

pd.plotting.andrews_curves(df, 'Class', color = ['blue', 'green'])

# this is a standard scatter plot

pd.plotting.scatter_matrix(df)


# ploltly plotting; this can be rendered in jupyterlab

index_vals = df['Class'].astype('category').cat.codes

fig = go.Figure(data=go.Splom(
               dimensions=[dict(label='Area', values=df['Area']),
                                             dict(label='MajorAxisLength', values=df['MajorAxisLength']),
                                             dict(label='MinorAxisLength', values=df['MinorAxisLength']),
                                             dict(label='Eccentricity', values=df['Eccentricity']),
                                             dict(label='ConvexArea', values=df['ConvexArea']),
                                             dict(label='Extent', values=df['Extent']),
                                             dict(label='Perimeter', values=df['Perimeter'])],
                showupperhalf=False, # remove plots on diagonal
                text=df['Class'],
                marker=dict(color=index_vals,
                            showscale=False, # colors encode categorical variables
                            line_color='white', line_width=0.5, opacity =0.25)
                ))


fig.update_layout(
    title='Raisin Data set',
    width=600,
    height=600,
)

fig.show()

## plolty plotting can be rendered in jupyterlab

## the above code is to plot scatter plot matrix and adrews curves

# the below code is used to do a histogram of standardized dataset comparison between 
# Kecimen and Besni for each dependent real variable.

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4,2, sharex = True)

axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

for n in range(0, len(axs)):
    axs[n].hist(transposed_label1[n], bins = 'auto')
    axs[n].hist(transposed_label2[n], bins = 'auto')
    
    






    
    


