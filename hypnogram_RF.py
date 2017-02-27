import pandas as pd
import numpy as np
from os import path
import json
from ast import literal_eval

from scipy import signal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from public_MAPE import score_function

# Extract new features from hypnograms and apply a Random Forest to it


def get_hypno(row):
    hypno = pd.Series(literal_eval(row)).astype(np.int8)
    return hypno


def get_duration_sleep(row):
    # Total sleep time: we don't take into account when the subject is awake (0)
    newRow = row[row.notnull()]
    sleep = (newRow != 0).sum()
    return sleep*30.0


def get_sleep_latency(row):
    # When the subject falls asleep
    cnt = 0.0
    for i, val in enumerate(row):
        if val == 0:
            cnt += 1
        else:
            return cnt*30.0


def get_sleep_efficiency(row):
    # Percentage of sleep time in comparison with time in bed
    newRow = row[row.notnull()]
    sleep_time = (newRow != 0).sum()
    time_bed = newRow.count()
    return float(sleep_time) / float(time_bed)


def get_perc_sleep_stage(row, id_stage):
    # Percentage of sleep stage "id_stage"
    newRow = row[row.notnull()]
    sleep_stage = (newRow == id_stage).sum()
    sleep_time = (newRow != 0).sum()
    return float(sleep_stage) / float(sleep_time)


def get_duration_sleep_stage(row, id_stage):
    # Duration of a sleep stage
    sleep_stage = (row == id_stage).sum()
    return sleep_stage*30.0


def get_periods_sleep_stage(row, id_stage):
    # Extract the mean and the max of each period with a succession of the same sleep stage
    periods = [] # periods is the duration of the sleep stage at each time
    cnt = 1 * (row[0] == id_stage) # cnt is incremented each time to get the period
    for i in range(1, len(row)):
        if row[i] == id_stage:
            if row[i-1] != id_stage:
                cnt = 1
            else:
                cnt += 1
        else:
            if row[i-1] == id_stage:
                periods.append(cnt)
                cnt = 0
    if cnt > 0: periods.append(cnt) # At the end we see if we have the desired sleep stage or not
    if len(periods) == 0: periods.append(0)
    return pd.Series({"S{}_MEAN".format(id_stage): 30*np.mean(periods),
                      "S{}_MAX".format(id_stage): 30*np.max(periods)})


def get_features_df(df):
    # Applies the whole preprocessing to the dataframe df
    sleep_stages = np.arange(5)
    hypno_list = df["HYPNOGRAM"]
    hypno = hypno_list.apply(get_hypno)
    # we replace the nan values (because the hypnograms don't have the same lengths) with -2
    # so they are not confused with -1 (missing values)
    hypno.fillna(-2, inplace=True)
    # then we fill the missing values with the previous valid value
    hypno.replace(-1, np.nan, inplace=True)
    hypno.fillna(method="ffill", axis=1, inplace=True)
    # we reset -2 as nan
    hypno.replace(-2, np.nan, inplace=True)
    # extraction of the features
    features_df = pd.DataFrame([])
    features_df["TST"] = hypno.apply(get_duration_sleep, axis=1)
    features_df["SE"] = hypno.apply(get_sleep_efficiency, axis=1)
    features_df["SLAT"] = hypno.apply(get_sleep_latency, axis=1)
    for id_stage in sleep_stages:
        features_df["S{}_PERC".format(id_stage)] = hypno.apply(get_perc_sleep_stage, args=(id_stage,), axis=1)
        features_df = features_df.merge(hypno.apply(get_periods_sleep_stage, args=(id_stage,), axis=1),
                                        left_index=True, right_index=True)
    return features_df


if __name__ == "__main__":
	print "Loading data"
	data_dir = "data"
	train = pd.read_csv(path.join(data_dir, "train_input.csv"), sep=';', index_col=0)
	test = pd.read_csv(path.join(data_dir, "test_input.csv"), sep=';', index_col=0)
	ages = pd.read_csv(path.join(data_dir, "train_output.csv"), sep=';', index_col=0)

	print "Extracting features from the data"
	features_train = get_features_df(train)
	features_test = get_features_df(test)

	print "Fitting the Random Forest estimator"
	scorer = make_scorer(score_function, greater_is_better=False)

	tuned_parameters = [{'n_estimators': [10, 20, 50, 100, 200, 500],
                     'min_samples_leaf': [1, 5, 10, 50, 100, 200, 500]}]

	X_train, y_train = features_train.values.astype(np.float64), ages.values.ravel()
	X_test = features_test.values.astype(np.float64)
	feature_names = list(features_train.columns)

	rf = RandomForestRegressor(oob_score=True, n_jobs=-1, random_state=42)
	reg = GridSearchCV(rf, tuned_parameters, cv=5, n_jobs=-1, verbose=1,
                       scoring=scorer) # Taking the oob_score is important to avoid overfitting

	reg.fit(X_train, y_train)

	print "best parameters"
	print reg.best_params_
	print "best score"
	print -reg.best_score_

	y_train_pred = reg.best_estimator_.oob_prediction_ # We shouldn't take reg.predict(X_train)!
	y_test_pred = reg.predict(X_test)

	print "Saving results"
	res_dir = "results"

	ages_train_pred = pd.DataFrame(np.round(y_train_pred), columns=["TARGET"], index=train.index)
	ages_train_pred.to_csv(path.join(res_dir, "train_output_hypnogram.csv"), sep=';')

	ages_test_pred = pd.DataFrame(np.round(y_test_pred), columns=["TARGET"], index=test.index)
	ages_test_pred.to_csv(path.join(res_dir, "test_output_hypnogram.csv"), sep=';')
