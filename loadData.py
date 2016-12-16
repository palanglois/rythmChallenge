import pandas as pd
import sklearn
from matplotlib.mlab import PCA
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt


data = pd.read_csv("train_input.csv", sep=';')
label = pd.read_csv("train_output.csv", sep=';')

data2 = data.iloc[0:-1,0:-2]
label2 = label.iloc[0:-1,1]

pls2 = PLSRegression()
pls2.fit(data2,label2)


