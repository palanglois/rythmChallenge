#!/usr/bin/python

import os
import shutil

#Import random
import random

#Import tensor flow
import tensorflow as tf
sess = tf.InteractiveSession()

class rythmCNN:

  def __init__(self,dimData,dimOutput,sizeHidden,batchSize,learningRate,nonLinearity,sizeSeg):

    #Setting parameters
    self.dimData = dimData
    self.dimOutput = dimOutput
    self.batchSize = batchSize
    self.learningRate = learningRate
    self.sizeHidden = sizeHidden
    self.name = 'hidden_'+str(sizeHidden)+'_lr_'+str(learningRate)+'_nl_'+nonLinearity+'_seg_'+str(sizeSeg)
    self.remotepath = '/home/pi/Documents/rythmProject'

    #Setting inner variables
    self.iterations = []
    
    #Allocating sizes for the data and the labels
    self.x = tf.placeholder(tf.float32, shape=[None, dimData])
    self.y_ = tf.placeholder(tf.float32, shape=[None, dimOutput])

    #Defining the model 

    #1st layer : fully connected
    self.W_fc1 = self.weight_variable([self.dimData,self.sizeHidden])
    self.b_fc1 = self.bias_variable([self.sizeHidden])
    if nonLinearity == 'sigmoid':
      self.h_fc1 = tf.nn.sigmoid(tf.matmul(self.x,self.W_fc1)+self.b_fc1)
    else:
      self.h_fc1 = tf.nn.relu(tf.matmul(self.x,self.W_fc1)+self.b_fc1)

    #Performing dropout
    self.keep_prob = tf.placeholder(tf.float32) #Proba to keep a neuron's output
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

    #2nd layer : fully connected
    self.W_fc2 = self.weight_variable([self.sizeHidden,self.sizeHidden])
    self.b_fc2 = self.bias_variable([self.sizeHidden])
    if nonLinearity == 'sigmoid':
      self.h_fc2 = tf.nn.sigmoid(tf.matmul(self.h_fc1_drop,self.W_fc2)+self.b_fc2)
    else:
      self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1_drop,self.W_fc2)+self.b_fc2)

    #Performing dropout
    self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

    #3rd layer : fully connected with softmax
    self.W_fc3 = self.weight_variable([self.sizeHidden,self.dimOutput])
    self.b_fc3 = self.bias_variable([self.dimOutput])
    self.y = tf.matmul(self.h_fc2_drop,self.W_fc3)+self.b_fc3

    #Defining the loss and accuracy
    self.divider = 1 if tf.argmax(self.y_,1) == 0 else tf.argmax(self.y_,1)
    self.accuracy = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(tf.argmax(self.y,1),tf.argmax(self.y_,1)),self.divider)))
    #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_) + 0.001*tf.nn.l2_loss(tf.nn.softmax(self.y)-self.y_))
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

    #Creating a training step
    self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

    #Create the summary variables
    tf.summary.scalar('Accuracy',self.accuracy)
    tf.summary.scalar('Loss',self.loss)
    
    #Creating a training writer
    self.merged = tf.summary.merge_all()
    if not os.path.exists('results'):
      os.makedirs('results')
    self.localpath = 'results/'+self.name+'/'
    if os.path.exists(self.localpath):
      shutil.rmtree(self.localpath)
    os.makedirs(self.localpath)
    self.localTrainPath = self.localpath+'train/'
    os.makedirs(self.localTrainPath)
    self.localValPath = self.localpath+'val/'
    os.makedirs(self.localValPath)
    self.train_writer = tf.summary.FileWriter(self.localTrainPath,sess.graph)
    self.val_writer = tf.summary.FileWriter(self.localValPath,sess.graph)

    #Performing the initialization in the back-end
    sess.run(tf.global_variables_initializer())

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial) 

  def bias_variable(self,shape):
    initial = tf.constant(1., shape=shape)
    return tf.Variable(initial)

  def sendFullSummary(self):
    #Saving the full result
    os.system('scp -P 50683 -r '+ self.localpath + ' pi@86.252.86.222:' + self.remotepath)

  def sendPartialSummary(self):
    #Saving the current result
    os.system('scp -P 50683 ' + self.localValPath + '*' + ' pi@86.252.86.222:' + self.remotepath)

  def reloadModel(self,ckptName):
    saver = tf.train.Saver()
    saver.restore(sess,ckptName)

  def train(self,nIter,npData,yLabel,xVal,yVal):
    
    #Doing nIter iterations
    for i in range(nIter):
      batchInd = random.sample(range(npData.shape[0]),self.batchSize)
      yBatch = yLabel[batchInd,:]
      xBatch = npData[batchInd,:]
      _, loss_val =  sess.run([self.train_step,self.loss], feed_dict={self.x: xBatch, self.y_: yBatch, self.keep_prob:0.5})
      summary_train = self.merged.eval(feed_dict={self.x: xBatch, self.y_: yBatch, self.keep_prob:1})
      summary_val = self.merged.eval(feed_dict={self.x: xVal, self.y_: yVal, self.keep_prob:1})
      self.train_writer.add_summary(summary_train, i)
      self.val_writer.add_summary(summary_val, i)

      if i % 50000 == 0:
        self.sendPartialSummary()

      if i % 10 == 0:
        train_accuracy = self.accuracy.eval(feed_dict={ self.x:xBatch, self.y_:yBatch, self.keep_prob:1.0})
        val_accuracy = self.accuracy.eval(feed_dict={ self.x:xVal, self.y_:yVal, self.keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print("step %d, evaluation accuracy %g"%(i, val_accuracy))
        #print "\ny"
        #print y.eval(feed_dict={ x:xBatch, y_:yBatch })
        #print "\ny_"
        #print y_.eval(feed_dict={ x:xBatch, y_:yBatch })
        print("step %d, loss %g"%(i, loss_val))
      self.iterations.append(loss_val)
    self.sendFullSummary()

