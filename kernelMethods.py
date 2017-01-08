import pandas as pd
import numpy as np
from os import path
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

plt.style.use("bmh")

data_dir = "data"

train = pd.read_csv(path.join(data_dir, "train_input.csv"), sep=';', index_col=0)
ages = pd.read_csv(path.join(data_dir, "train_output.csv"), sep=';', index_col=0)
