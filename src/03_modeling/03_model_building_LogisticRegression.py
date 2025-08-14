import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

df = pd.read_pickle("../../data/interim/01_make_dataset.pkl")
x_train = pd.read_pickle("../../data/interim/x_train.pkl")
x_test = pd.read_pickle("../../data/interim/x_test.pkl")
y_train = pd.read_pickle("../../data/interim/y_train.pkl")
y_test = pd.read_pickle("../../data/interim/y_test.pkl")

df.head()
x_train
x_test
y_train
y_test


## Logistic Regression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train, y_train)


y_pred = classifier_lr.predict(x_test)

y_pred

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


result = pd.DataFrame(
    [["Logistic Regression", acc, f1, prec, rec]],
    columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"],
)

result

cm = confusion_matrix(y_test, y_pred)
print(cm)


## Cross Validatino
# We can Evaluate the performance of our calsification Model

accuracies = cross_val_score(estimator=classifier_lr, X=x_train, y=y_train, cv=10)

print("Accuricy is {:.2f} %".format(accuracies.mean() * 100))
# 97.81 %
print("Std Deviation is {:.2f} %".format(accuracies.std() * 100))
# 1.98 %
