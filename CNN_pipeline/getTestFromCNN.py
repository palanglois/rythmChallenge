import numpy as np
import pandas as pd

def getTestFromCNN(cnn,csvName):
  data = pd.read_csv("../test_input.csv", sep=';')
  npData = np.array([data.iloc[i,2:-1].as_matrix() for i in range(data.shape[0])])
  xTest_fil = butter_bandpass_filter(npData, lowFreq, highFreq, sampling_freq, order=butter_order)
  fakeY = np.zeros((249,90))
  xTestFinal, yTestFinal = augmentData(xTest_fil,fakeY,sizeSeg,lowFreq,highFreq)
  xTestFinal = normalizeData(xTestFinal)
  yTestAges = np.argmax(cnn.y.eval(feed_dict={ cnn.x:xTestFinal, cnn.keep_prob:1.0}),1)
  dimData = xTest_fil.shape[1]
  nFactor = dimData // sizeSeg
  yReal = averageFun(yTestAges,nFactor)
  np.savetxt(csvName, np.transpose(yReal), delimiter=",",fmt='%1.0f')


def averageFun(x,n):
  y = []
  for i in range(len(x)//n):
    accu = 0
    for j in range(n):
      accu = accu + x[i*n+j]
    y.append(float(accu)/n)
  return y

