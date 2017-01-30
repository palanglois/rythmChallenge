#!/usr/bin/python

#Load matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Import Numpy
import numpy as np

#Import random
import random

#Import tensor flow
import tensorflow as tf
sess = tf.InteractiveSession()

#import pandas and load the data
import pandas as pd
#All data data
data = pd.read_csv("train_input.csv", sep=';')
npData = np.array([data.iloc[i,2:-1].as_matrix() for i in range(data.shape[0])])
label = pd.read_csv("train_output.csv", sep=';')
npLabel = np.array([label.iloc[i,1] for i in range(label.shape[0])])


#Retrieving the dimension
dimData= len(npData[0])
dimOutput = 90

#Batch size
batchSize = 20

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
yLabel = yLabel[0:beginVal,:]
npData = npData[0:beginVal,:]

#Allocating sizes for the data and the labels
x = tf.placeholder(tf.float32, shape=[None, dimData])
y_ = tf.placeholder(tf.float32, shape=[None, dimOutput])

#Defining weights
W = tf.Variable(tf.zeros([dimData,dimOutput]))
b = tf.Variable(tf.zeros([dimOutput]))

#Defining the model (1 layer)
y = tf.nn.softmax(tf.matmul(x,W) + b)

#Defining the loss
divider = 1 if tf.argmax(y_,1) == 0 else tf.argmax(y_,1)
accuracy = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(tf.argmax(y,1),tf.argmax(y_,1)),divider)))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))


#Defining the learning rate
learningRate = 1e-2

#Creating a training step
train_step = tf.train.AdamOptimizer(learningRate).minimize(loss)

#Determining the number of correct predictions
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#Averaging the number of correct predictions
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Performing the initialization in the back-end
sess.run(tf.global_variables_initializer())

#Doing 1000 iterations
iterations = []
for i in range(1000):
  batchInd = random.sample(range(dimOutput),batchSize)
  #batchInd = [np.random.randint(0,90) for j in range(batchSize)]
  yBatch = yLabel[batchInd,:]
  xBatch = npData[batchInd,:]
  if i % 10 == 0:
    train_accuracy = accuracy.eval(feed_dict={ x:xBatch, y_:yBatch })
    val_accuracy = accuracy.eval(feed_dict={ x:xVal, y_:yVal})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    print("step %d, evaluation accuracy %g"%(i, val_accuracy))
  #print "\ny"
  #print y.eval(feed_dict={ x:xBatch, y_:yBatch })
  #print "\ny_"
  #print y_.eval(feed_dict={ x:xBatch, y_:yBatch })
  _, loss_val =  sess.run([train_step,loss], feed_dict={x: xBatch, y_: yBatch})
  iterations.append(loss_val)

