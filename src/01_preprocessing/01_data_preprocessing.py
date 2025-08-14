import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
from sklearn.model_selection import train_test_split


# Importing data into dataframe
df = pd.read_csv("./data.csv")
df.head()

############################
# Dealing with NaN values
############################

# NAN values present? (Yes/No)
df.isnull().values.any()

# NAN values, How may?
df.isnull().values.sum()

# column with null values
df.columns[df.isnull().any()]

# count of column with null values
len(df.columns[df.isnull().any()])

# How many values are not NAN?
df["Unnamed: 32"].count()
# '0' Hence all values are null

# We can remove this column...Unnamed: 32
print(df["Unnamed: 32"])

# removing NaN Valued colimn from df
df = df.drop(columns="Unnamed: 32")
df.shape


############################
# Dealing with Categorical Data
############################

df.select_dtypes(include="object").columns
df["diagnosis"].unique()
df["diagnosis"].nunique()
# We can covert data into binary values
# in numarical values for easy handling.

# One hot encoding
df = pd.get_dummies(data=df, drop_first=True)

df["diagnosis_M"].unique()
df["diagnosis_M"] = df["diagnosis_M"].astype(int)
df.head()

############################
# CountPlot
############################

sns.countplot(x="diagnosis_M", data=df)
plt.show()

# B (0) values
(df.diagnosis_M == 0).sum()
# M (1) values
(df.diagnosis_M == 1).sum()


############################
## Corelation Matrix & Heatmap
############################

df_cr = df.drop(columns="diagnosis_M")
df_cr.head()

df_cr.corrwith(df["diagnosis_M"]).plot.bar(
    figsize=(20, 30), title="Correlated with diagnosis_M", rot=45, grid=True
)

# Corelation Matrix
corr = df.corr()
corr

# Heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True)

############################
# Export dataframe(df) into pickle
############################

pd.to_pickle("./../../data/interim/01_make_dataset.pkl")
