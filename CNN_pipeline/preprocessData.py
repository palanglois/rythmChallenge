from scipy.signal import butter, lfilter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y


#n is the size of one subsample
def augmentData(x,y,n):
  nbData = x.shape[0]
  dimData = x.shape[1]
  #For one data, we make nFactor data
  nFactor = dimData // n
  newX = []
  newY = []
  for i in range(nbData):
    for j in range(nFactor):
      subX = x[i,j*n:(j+1)*n]
      subY = y[i]
      newX.append(subX)
      newY.append(subY)
  return np.array(newX), np.array(newY)


