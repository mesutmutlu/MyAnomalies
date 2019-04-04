from sklearn.ensemble import  AdaBoostRegressor
from earthquake.read_file import get_features
from sklearn.model_selection import  RepeatedKFold
from sklearn.metrics import mean_absolute_error
import pandas as pd

kfold = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2652124)
cv_train_scores = []
cv_train_loss = []
cv_val_scores = []
cv_val_loss = []

X_train, y_train, X_test = get_features()

n_grid = 10
final_scores = pd.DataFrame(index=[i for i in range(n_grid)])
splits = kfold.split(X_train, y_train)
cv_scores = pd.DataFrame(index=[i for i in range(10)])
i =0
for train, val in splits:
    rlgb = AdaBoostRegressor()

    rlgb.fit(X_train.iloc[train].values, y_train.iloc[train].values.ravel())

    pred = rlgb.predict(X_train.iloc[val].values)
    score = mean_absolute_error(y_train.iloc[val].values, pred)
    cv_scores.loc[i, "score"] = score
    i += 1
print(cv_scores["score"].mean())