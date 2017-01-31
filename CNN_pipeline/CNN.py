#!/usr/bin/python

#Import random
import random

#Import tensor flow
import tensorflow as tf
sess = tf.InteractiveSession()

class rythmCNN:

  def __init__(self,dimData,dimOutput,sizeHidden,batchSize,learningRate):

    #Setting parameters
    self.dimData = dimData
    self.dimOutput = dimOutput
    self.batchSize = batchSize
    self.learningRate = learningRate
    self.sizeHidden = sizeHidden

    #Allocating sizes for the data and the labels
    self.x = tf.placeholder(tf.float32, shape=[None, dimData])
    self.y_ = tf.placeholder(tf.float32, shape=[None, dimOutput])
    
    #Defining weights
    self.W = tf.Variable(tf.zeros([self.dimData,self.dimOutput]))
    self.b = tf.Variable(tf.zeros([self.dimOutput]))

    #Defining the model (1 input layer)
    self.y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)

    #Defining the loss and accuracy
    self.divider = 1 if tf.argmax(self.y_,1) == 0 else tf.argmax(self.y_,1)
    self.accuracy = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(tf.argmax(self.y,1),tf.argmax(self.y_,1)),self.divider)))
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

    #Creating a training step
    self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

    #Performing the initialization in the back-end
    sess.run(tf.global_variables_initializer())

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial) 

  def bias_variable(self,shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def train(self,nIter,npData,yLabel,xVal,yVal):
    
    self.iterations = []
    #Doing nIter iterations
    for i in range(nIter):
      batchInd = random.sample(range(npData.shape[0]),self.batchSize)
      yBatch = yLabel[batchInd,:]
      xBatch = npData[batchInd,:]
      if i % 10 == 0:
        train_accuracy = self.accuracy.eval(feed_dict={ self.x:xBatch, self.y_:yBatch })
        val_accuracy = self.accuracy.eval(feed_dict={ self.x:xVal, self.y_:yVal})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print("step %d, evaluation accuracy %g"%(i, val_accuracy))
      #print "\ny"
      #print y.eval(feed_dict={ x:xBatch, y_:yBatch })
      #print "\ny_"
      #print y_.eval(feed_dict={ x:xBatch, y_:yBatch })
      _, loss_val =  sess.run([self.train_step,self.loss], feed_dict={self.x: xBatch, self.y_: yBatch})
      self.iterations.append(loss_val)

