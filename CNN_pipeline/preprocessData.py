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
def augmentData(x,y,n,lowFreq,highFreq):
  nbData = x.shape[0]
  dimData = x.shape[1]
  #For one data, we make nFactor data
  nFactor = dimData // n
  newX = []
  newY = []
  for i in range(nbData):
    for j in range(nFactor):
      subX = np.log(np.absolute(np.fft.fft(x[i,j*n:(j+1)*n])))
      subY = y[i]
      freqs = np.fft.fftfreq(len(subX))*250
      ind = np.where(np.logical_and(freqs<highFreq,freqs>lowFreq))
      subX = subX[ind]
      newX.append(subX)
      newY.append(subY)
  return np.array(newX), np.array(newY)

def normalizeData(x):
  xMean = np.mean(x,0)
  xDev = np.std(x,0)
  xDev[xDev == 0] = 1
  return np.divide((x - xMean), xDev)
