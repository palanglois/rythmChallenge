import pandas as pd
import sklearn
from matplotlib.mlab import PCA
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import numpy as np

def score_function(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

data = pd.read_csv("train_input.csv", sep=';')
label = pd.read_csv("train_output.csv", sep=';')

data2 = data.iloc[0:-1,0:-2]
label2 = label.iloc[0:-1,1]

pls2 = PLSRegression(n_components=10)
pls2.fit(data2,label2)

label_pred = pls2.predict(data2).reshape((label2.shape[0],))

print score_function(label2,label_pred)

