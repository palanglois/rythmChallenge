#!/bin/python

#Defining parameters
lowFreq = 0.5 #Hz
highFreq = 100 #Hz
sampling_freq = 250 #Hz
butter_order = 2 #Order of the butterworth filter
dimOutput = 90 #We predict ages from 0 to 89 years old
sizeHidden = 10 #Size of the hidden layers
batchSize = 400 #Size of the batch for training
learningRate = 0.05 #Setting learning rate

#Loading the data
print('\nLoading the data...')
from loadData import *
print('Data loaded !\n')

#Filtering
print('Filtering the data...')
from preprocessData import *
xTrain_fil = butter_bandpass_filter(xTrain, lowFreq, highFreq, sampling_freq, order=butter_order)
yTrain_fil = butter_bandpass_filter(yTrain, lowFreq, highFreq, sampling_freq, order=butter_order)
xVal_fil = butter_bandpass_filter(xVal, lowFreq, highFreq, sampling_freq, order=butter_order)
yVal_fil = butter_bandpass_filter(yVal, lowFreq, highFreq, sampling_freq, order=butter_order)
print('Data filtered !\n')

#Augmenting the data
print('Augmenting the data...')
xTrainFinal, yTrainFinal = augmentData(xTrain_fil,yTrain_fil,1250)
xValFinal, yValFinal = augmentData(xVal_fil,yVal_fil,1250)
print('Data augmented !\n')
print('The data is now ready for training...\n')

#Creating the CNN graph
from CNN import *
cnn = rythmCNN(xTrainFinal.shape[1],dimOutput,sizeHidden,batchSize,learningRate)
cnn.train(10000,xTrainFinal,yTrainFinal, xValFinal, yValFinal)

