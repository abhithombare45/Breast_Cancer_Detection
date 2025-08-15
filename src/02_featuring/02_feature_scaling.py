import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_pickle("../../data/interim/01_make_dataset.pkl")

df.head()

############################
## Splitting dataset TRAIn & TEST dataset
############################

# matrix of feature / independent variable

x = df.iloc[:, 1:-1].values
x.shape

# Target Variable / dependent variable
y = df.iloc[:, -1].values
y.shape

# Import sklearn(scikit-learn) for spliting data
# into train and Test dataset
sktt = train_test_split
x_train, x_test, y_train, y_test = sktt(x, y, test_size=0.2, random_state=0)

x_train.shape
x_train
x_test.shape
x_test
y_train.shape
y_train
y_test.shape
y_test


############################
## Feature Scaling
############################

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train  # x_train.shape
x_test  # x_test.shape


############################
## Exporting  Train & Test data
############################

pd.to_pickle(x_train, "../../data/interim/x_train.pkl")
pd.to_pickle(x_test, "../../data/interim/x_test.pkl")
pd.to_pickle(y_train, "../../data/interim/y_train.pkl")
pd.to_pickle(y_test, "../../data/interim/y_test.pkl")