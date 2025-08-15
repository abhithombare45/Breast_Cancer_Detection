import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


x_train = pd.read_pickle("../../data/interim/x_train.pkl")
x_test = pd.read_pickle("../../data/interim/x_test.pkl")
y_train = pd.read_pickle("../../data/interim/y_train.pkl")
y_test = pd.read_pickle("../../data/interim/y_test.pkl")

classifier = LogisticRegression(
    penalty="l2",
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=0,
    solver="lbfgs",
    max_iter=100,
    multi_class="deprecated",
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


result = pd.DataFrame(
    [["Final Logistic Regression", acc, f1, prec, rec]],
    columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"],
)

result_LR_RF = pd.read_pickle("../../data/interim/result_LR_RF.pkl")
print(result_LR_RF)

result_final = pd.concat([result, result_LR_RF], ignore_index=True)
print(result_final)
print(result)

## Cross validation

accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)

print("Accuricy is {:.2f} %".format(accuracies.mean() * 100))
# 97.81
print("Std Deviation is {:.2f} %".format(accuracies.std() * 100))
# 1.98 %
