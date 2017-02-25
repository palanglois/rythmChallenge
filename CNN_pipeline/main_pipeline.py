#!/bin/python

#Defining parameters
lowFreq = 5 #Hz
highFreq = 20 #Hz
sampling_freq = 250 #Hz
butter_order = 2 #Order of the butterworth filter
dimOutput = 90 #We predict ages from 0 to 89 years old
sizeHidden = 200 #Size of the hidden layers
batchSize = 4000 #Size of the batch for training
learningRate = 0.00001 #Setting learning rate
sizeSeg = 1250*4 #Size of the segmented data
nonLinearity = 'sigmoid'

#Loading the data
print('\nLoading the data...')
from loadData import *
print('Data loaded !\n')

#Filtering
print('Filtering the data...')
from preprocessData import *
xTrain_fil = butter_bandpass_filter(xTrain, lowFreq, highFreq, sampling_freq, order=butter_order)
xVal_fil = butter_bandpass_filter(xVal, lowFreq, highFreq, sampling_freq, order=butter_order)
print('Data filtered !\n')

#Augmenting the data
print('Augmenting the data...')
xTrainFinal, yTrainFinal = augmentData(xTrain_fil,yTrain,sizeSeg,lowFreq,highFreq)
xValFinal, yValFinal = augmentData(xVal_fil,yVal,sizeSeg,lowFreq,highFreq)
print('Data augmented !\n')

#Normalizing the data
print('Normalizing the data...')
xTrainFinal = normalizeData(xTrainFinal)
xValFinal = normalizeData(xValFinal)
print('Data normalized !\n')
print('The data is now ready for training...\n')

#Creating the CNN graph
print('Loading tensorflow...\n')
from CNN import *
print('\n\nLoading the neural network...')
cnn = rythmCNN(xTrainFinal.shape[1],dimOutput,sizeHidden,batchSize,learningRate,nonLinearity,sizeSeg)

#print('Training !')
#cnn.train(100000,xTrainFinal,yTrainFinal, xValFinal, yValFinal)
print('Reloading weights')
cnn.reloadModel('model100k.ckpt')
print('If no error, model is ready')
