import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
classifier_rf = RandomForestClassifier(random_state=0)
classifier_rf.fit(x_train, y_train)

y_pred = classifier_rf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


model_result = pd.DataFrame(
    [["Random Forest", acc, f1, prec, rec]],
    columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"],
)

result = pd.read_pickle("../../data/interim/result_LR.pkl")
print(result)

result = pd.concat([result, model_result], ignore_index=True)
print(result)

pd.to_pickle(result, "../../data/interim/result_LR_RF.pkl")
print(result)

# checking Confustion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# We got same accuracy for both the models, also 4 wrong prediction for both models
# Now to select Final Model we need to
# Cross validate Random Forest model.

accuracies = cross_val_score(estimator=classifier_rf, X=x_train, y=y_train, cv=10)

print("Accuricy is {:.2f} %".format(accuracies.mean() * 100))
# 97.81
print("Std Deviation is {:.2f} %".format(accuracies.std() * 100))
# 1.98 %

# Loading modified Classifier data
joblib.dump(classifier_rf, "./../../data/interim/classifier_rf_model.pkl")
