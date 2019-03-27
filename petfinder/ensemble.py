from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import sys
from datetime import  datetime
from sklearn.model_selection import train_test_split
from matplotlib import  pyplot as plt

from sklearn.metrics import cohen_kappa_score, make_scorer

def kappa_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    return 1 - cohen_kappa_score(y_true, y_pred, weights="quadratic")

def kappa_metric(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
print(cohen_kappa_score)

train = pd.read_csv(r"C:\Users\dtmemutlu\Downloads\train.csv")
test = pd.read_csv(r"C:\Users\dtmemutlu\Downloads\test.csv")

x_indep = train.drop("AdoptionSpeed", axis=1)
y_dep = train[["AdoptionSpeed"]]


X_train, X_val, y_train, y_val = train_test_split(x_indep, y_dep, test_size=0.25, random_state= 42, stratify=y_dep)

y_train.hist()
#plt.show()
y_val.hist()
#plt.show()



lgb = LGBMClassifier(boosting= 'gbdt',
          max_depth= 11,
          objective='multiclass',
          num_leaves= 350,
          learning_rate= 0.01,
          bagging_fraction= 0.85,
          feature_fraction= 0.8,
          min_split_gain= 0.01,
          min_child_samples= 75,
          min_child_weight= 0.1,
          verbosity= -1,
          data_random_seed= 3,
          n_jobs=2,
          #lambda_l2= 0.05,
          class_weight='balanced')
#lgb.fit(X_train, y_train)
xgb = XGBClassifier(booster= 'dart',
          max_depth= 9,
          learning_rate= 0.01,
          gamma= 0.01,
          min_child_weight= 1,
          verbosity= -1,
          random_state= 3,
          verbose_eval= False,
          n_jobs=2)
#xgb.fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=11,
                            min_samples_split=500,
                            min_samples_leaf=500,
                            random_state=3,
                            class_weight='balanced')
#dt.fit(X_train, y_train)
print(datetime.now())
ada = AdaBoostClassifier(base_estimator=lgb)
ada.fit(X_train, y_train["AdoptionSpeed"], )
pred1 = ada.predict(X_val)
print(datetime.now())
ada = AdaBoostClassifier(base_estimator=xgb)
ada.fit(X_train, y_train["AdoptionSpeed"].values.ravel())
pred2 = ada.predict(X_val)
print(datetime.now())
ada = AdaBoostClassifier(base_estimator=dt)
ada.fit(X_train, y_train["AdoptionSpeed"].values.ravel())
pred3 = ada.predict(X_val)
print(datetime.now())
print(pred3)



ada_f = AdaBoostClassifier(base_estimator=dt)
ada.fit()


