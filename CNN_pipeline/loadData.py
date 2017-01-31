#!/bin/python

import cPickle
import numpy as np

#Loading Training data
xTrain_file = open('xTrain.pkl', 'rb')
xTrain = np.array(cPickle.load(xTrain_file))

yTrain_file = open('yTrain.pkl','rb')
yTrain = np.array(cPickle.load(yTrain_file))

#Loading Validation data
xVal_file = open('xVal.pkl', 'rb')
xVal = np.array(cPickle.load(xVal_file))

yVal_file = open('yVal.pkl','rb')
yVal = np.array(cPickle.load(yVal_file))

