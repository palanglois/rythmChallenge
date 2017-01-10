# coding: utf-8

import pandas as pd
import numpy as np
from os import path
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from public_MAPE import score_function


plt.style.use("bmh")


def get_hypno(row):
    hypno = pd.Series(literal_eval(row)).astype(np.int8)
    return hypno


def get_duration_sleep(row):
    sleep = row.notnull().sum()
    return sleep*30


def get_duration_deep_sleep(row):
    deep_sleep = (row == 3).sum()
    return deep_sleep*30


def get_periods_deep_sleep(row):
    periods = []
    cnt = 1 * (row[0] == 3)
    for i in range(1, len(row)):
        if row[i] == 3:
            if row[i-1] != 3:
                cnt = 1
            else:
                cnt += 1
        else:
            if row[i-1] == 3:
                periods.append(cnt)
                cnt = 0
    if cnt > 0: periods.append(cnt)
    return pd.Series({"AVERAGE_DEEP_SLEEP": 30*np.mean(periods), "MAX_DEEP_SLEEP": 30*np.max(periods)})


def deal_with_hypno(train):
    train_hypno = pd.DataFrame([])
    hypno = train["HYPNOGRAM"].apply(get_hypno)
    # we replace the nan values (because the hypnograms don't have the same lengths) with -2
    # so they are not confused with -1 (missing values)
    hypno.fillna(-2, inplace=True)
    # then we fill the missing values with the previous valid value
    hypno.replace(-1, np.nan, inplace=True)
    hypno.ffill(axis=1, inplace=True)
    # we reset -2 as nan
    hypno.replace(-2, np.nan, inplace=True)
    train_hypno["SLEEP_TIME"] = hypno.apply(get_duration_sleep, axis=1)
    train_hypno["DEEP_SLEEP_TIME"] = hypno.apply(get_duration_deep_sleep, axis=1)
    train_hypno = train_hypno.merge(hypno.apply(get_periods_deep_sleep, axis=1), left_index=True, right_index=True)
    return train_hypno


if __name__ == "__main__":
    print("Processing the hypnogram")
    data_dir = "data"
    train = pd.read_csv(path.join(data_dir, "train_input.csv"), sep=';', index_col=0)
    ages = pd.read_csv(path.join(data_dir, "train_output.csv"), sep=';', index_col=0)
    train_hypno = deal_with_hypno(train)

    data = train_hypno.values
    ages = ages.values.ravel()

    # pca = PCA(n_components=10, random_state=42)
    # pca.fit(data, ages)
    # data = pca.transform(data, ages)

    X_train, X_test, y_train, y_test = train_test_split(data, ages, test_size=0.3, random_state=42)

    print("Predicting with the data of the hypnogram")
    # reg = SVR(kernel="linear")
    reg = Lasso(random_state=42)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print(score_function(y_test, pred))

    plt.plot(y_test)
    plt.plot(pred)
    plt.show()
