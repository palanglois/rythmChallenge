import pandas as pd
import numpy as np
from os import path
import json
from ast import literal_eval

from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from public_MAPE import score_function

# We take the results from the ANN and the Random Forest and use a Linear Regression

if __name__ == "__main__":
	print "Loading data and results"
	data_dir = "data"
	res_dir = "results"
	ages = pd.read_csv(path.join(data_dir, "train_output.csv"), sep=';', index_col=0)
	train_hypno = pd.read_csv(path.join(res_dir, "train_output_hypnogram.csv"), sep=";", index_col=0)
	train_EEG = pd.read_csv(path.join(res_dir, "train_output_EEG.csv"), sep=";", index_col=0)
	test_hypno = pd.read_csv(path.join(res_dir, "test_output_hypnogram.csv"), sep=";", index_col=0)
	test_EEG = pd.read_csv(path.join(res_dir, "test_output_EEG.csv"), sep=";", index_col=0)
	test_EEG.index = test_hypno.index

	X_train = pd.concat([train_EEG, train_hypno], axis=1).values
	X_test = pd.concat([test_EEG, test_hypno], axis=1).values
	y_train = ages.values.ravel()

	print "Fitting the Linear estimator"
	reg = LinearRegression(fit_intercept=True, normalize=True)
	reg.fit(X_train, y_train)
	y_train_pred = reg.predict(X_train)
	print "Score: "
	print "With ANN:           ", score_function(y_train, X_train[:,0])
	print "With Random Forest: ", score_function(y_train, X_train[:,1])
	print "Wiht both:          ", score_function(y_train, y_train_pred)
	y_test = reg.predict(X_test)
	ages_pred = pd.DataFrame(np.round(y_test), columns=["TARGET"], index=test_hypno.index)
	ages_pred.to_csv(path.join(res_dir, "test_output.csv"), sep=';')
