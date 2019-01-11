
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import sys

from sklearn.base import BaseEstimator
from petfinder.preprocessing import prepare_data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from time import time
from numpy.random import random_integers #integers
from numpy.random import random_sample #floats
import pandas as pd
import scipy as sp
import random
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from random import choice
from sklearn.model_selection import RepeatedStratifiedKFold
from petfinder.get_explore import read_data


def run_cv_model(model, X, y, params, metric):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    folds = rskf.split(X, y)
    fold_scores = []

    for train_index, val_index in folds:
        print(train_index, val_index)
    pass


if __name__ == "__main__":

    train, test = read_data()
    x_train, y_train, x_test, id_test =  prepare_data(train, test)
    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
    run_cv_model("model", x_train, y_train, "params", kappa_scorer)