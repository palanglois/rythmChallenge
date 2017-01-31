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

    #Defining the model 

    #1st layer : fully connected
    self.W_fc1 = self.weight_variable([self.dimData,self.sizeHidden])
    self.b_fc1 = self.bias_variable([self.sizeHidden])
    self.h_fc1 = tf.nn.relu(tf.matmul(self.x,self.W_fc1)+self.b_fc1)

    #Performing dropout
    self.keep_prob = tf.placeholder(tf.float32) #Proba to keep a neuron's output
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

    #2nd layer : fully connected with softmax
    self.W_fc2 = self.weight_variable([self.sizeHidden,self.dimOutput])
    self.b_fc2 = self.bias_variable([self.dimOutput])
    self.y = tf.nn.softmax(tf.matmul(self.h_fc1_drop,self.W_fc2)+self.b_fc2)

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
      _, loss_val =  sess.run([self.train_step,self.loss], feed_dict={self.x: xBatch, self.y_: yBatch, self.keep_prob:0.5})
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

