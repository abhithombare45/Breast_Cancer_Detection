import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import RandomizedSearchCV

# randomized Search to Find the best parameters (Logestic Regression)
parameters = {
    "penalty": ["l1", "l2", "elasticnet", None],
    "C": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
}

parameters
# Load trained model
classifier_lr = joblib.load("./../../data/interim/classifier_lr_model.pkl")
classifier_lr
# Load x_train, y_train
x_train = pd.read_pickle("../../data/interim/x_train.pkl")
y_train = pd.read_pickle("../../data/interim/y_train.pkl")


random_search = RandomizedSearchCV(
    estimator=classifier_lr,
    param_distributions=parameters,
    n_iter=10,
    scoring="roc_auc",
    n_jobs=-1,
    cv=10,
    verbose=3,
)

random_search.fit(x_train, y_train)

random_search.best_score_
random_search.best_params_
random_search.best_estimator_

print("Best Parameters:", random_search.best_params_)
print("Best ROC AUC:", random_search.best_score_)
print("Best Model:", random_search.best_estimator_)

# Final Model
