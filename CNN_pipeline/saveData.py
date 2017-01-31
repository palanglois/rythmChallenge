#!/usr/bin/python

#Import Numpy
import numpy as np

#import pandas and load the data
import pandas as pd

#Import pickle for serialization
import cPickle 

#Loading all available data
data = pd.read_csv("train_input.csv", sep=';')
npData = np.array([data.iloc[i,2:-1].as_matrix() for i in range(data.shape[0])])
label = pd.read_csv("train_output.csv", sep=';')
npLabel = np.array([label.iloc[i,1] for i in range(label.shape[0])])

#Retrieving the dimension
dimData= len(npData[0])
dimOutput = 90

#Batch size
batchSize = 40

#Setting up the labels to the right format
yLabel = []
for label in npLabel:
  newLabel = np.zeros([dimOutput])
  newLabel[label] = 1
  yLabel.append(newLabel)
yLabel = np.array(yLabel)

#Evaluation data
percentageVal = 0.1
beginVal = int(len(npData)*(1-percentageVal))
yVal = yLabel[beginVal:,:]
xVal = npData[beginVal:,:]
#Training data
yTrain = yLabel[0:beginVal,:]
xTrain = npData[0:beginVal,:]

#Saving the data to files

#xTrain_file = open('xTrain.pkl', 'wb')
#cPickle.dump(xTrain,xTrain_file,-1)

yTrain_file = open('yTrain.pkl', 'wb')
cPickle.dump(yTrain,yTrain_file,-1)

xVal_file = open('xVal.pkl', 'wb')
cPickle.dump(xVal,xVal_file,-1)

yVal_file = open('yVal.pkl', 'wb')
cPickle.dump(yVal,yVal_file,-1)

