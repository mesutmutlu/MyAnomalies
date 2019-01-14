# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Albert Thomas <albert.thomas@telecom-paristech.fr>
# License: BSD 3 clause

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
# from joblib import parallel_backend
# from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from petfinder.get_explore import read_data


def report(df, alg, best_est, perf, est, results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            #print("Model with rank: {0}".format(i))
            #print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            #      results['mean_test_score'][candidate],
            #      results['std_test_score'][candidate]))
            #print("Parameters: {0}".format(results['params'][candidate]))
            #print("")
            df.loc[len(df)] = [alg, best_est, perf, est, format(i), results['mean_test_score'][candidate],results['std_test_score'][candidate],results['params'][candidate]]
    return df


def iterate_by_randomsearch(train_x, train_y):
    classifiers = [
        # (AdaBoostClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                           'learning_rate': sp.stats.uniform(0.0001, 1)  }),
        # (BaggingClassifier(),{"n_estimators": sp.stats.randint(25, 100),
        #                                      "max_features": sp.stats.randint(1, 7),
        #                                      "bootstrap": [True, False],
        #                                      "bootstrap_features": [True, False],
        #                                         }),
        # (ExtraTreesClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                             "max_depth": sp.stats.randint(3, 10),
        #                             "max_features": sp.stats.randint(1, 7),
        #                             "min_samples_split": sp.stats.randint(2, 11),
        #                             "bootstrap": [True, False],
        #                             "criterion": ["gini", "entropy"]}),
        # (RandomForestClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                             "max_depth": sp.stats.randint(3, 10),
        #                             "max_features": sp.stats.randint(1, 7),
        #                             "min_samples_split": sp.stats.randint(2, 11),
        #                             "bootstrap": [True, False],
        #                             "criterion": ["gini", "entropy"]}),
        # (PassiveAggressiveClassifier(), {"max_iter":sp.stats.randint(0, 1230),
        #                                  "tol": sp.stats.uniform(0.0001, 0.05)}),
        # (RidgeClassifier(), {"max_iter":sp.stats.randint(0, 2000),
        #                      "tol": sp.stats.uniform(0.0001, 0.05),
        #                      "solver":["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}),
        # (SGDClassifier(), {"max_iter":sp.stats.randint(0, 2000),
        #                       "tol": sp.stats.uniform(0.0001, 0.05),
        #                    "loss":["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        #                     "penalty":["none", "l2", "l1", "elasticnet"]}),
        # (KNeighborsClassifier(), {"n_neighbors":sp.stats.randint(1, 50),
        #                           "algorithm":["ball_tree", "kd_tree", "brute"],
        #                           "leaf_size":sp.stats.randint(20,100),
        #                           "p":[1,2]}),
        # (DecisionTreeClassifier(), {          "max_depth": sp.stats.randint(3, 10),
        #                                       "max_features": sp.stats.randint(1, 7),
        #                                       "min_samples_split": sp.stats.randint(2, 11),
        #                                       "criterion": ["gini", "entropy"]}),
        # (QuadraticDiscriminantAnalysis(), {"tol":sp.stats.uniform(1e-5, 1e-2)}),
        # (xgb.XGBClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                             "max_depth": sp.stats.randint(3, 30)}),
        (lgb.LGBMClassifier(), {'num_leaves': sp.stats.randint(25, 330),
              #'bagging_fraction': sp.stats.uniform(0.4, 0.9),
              'learning_rate': sp.stats.uniform(0.001, 0.5),
              #'min_data': sp.stats.randint(50,700),
              #'is_unbalance': [True, False],
              #'max_bin': sp.stats.randint(3,25),
              'boosting_type' : ['gbdt', 'dart'],
              #'bagging_freq': sp.stats.randint(3,35),
              'max_depth': sp.stats.randint(3,15),
              #'feature_fraction': sp.stats.uniform(0.4, 0.9),
              #'lambda_l1': sp.stats.randint(0,45),
              #'objective': 'multiclass',
                                "n_jobs":[-1],} )
    ]

    df = pd.DataFrame(columns=['alg', 'best_estimator', 'perf', 'est','rank','mean','std', 'parameters'])
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    folds = rskf.split(train_x, train_y)
    print(folds)
    for clf in classifiers:
        print(type(clf[0]).__name__, "started at", datetime.now())
        n_iter=10
        kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
        random_search = RandomizedSearchCV(clf[0], param_distributions=clf[1], cv=folds, scoring=kappa_scorer, n_jobs=2)
        start = time()
        random_search.fit(train_x, train_y)

        df = report(df, type(clf[0]).__name__, random_search.best_estimator_ ,  time() - start, n_iter, random_search.cv_results_)
        #print(type(clf[0]).__name__, "ended at", datetime.now())
    #print(df.sort_values(by="mean", ascending=False))
    best = df['mean'].idxmax()

    est = df.loc[best, ['best_estimator', "mean"]]
    #print(df.sort_values(by="mean", ascending=False)["best_estimator"][:5])
    i = 1
    clfs = []
    means = []
    for idx, clf in df.sort_values(by="mean", ascending=False)[:5].iterrows():
        #print(clf["best_estimator"])
        clfs.append ((str(i)+type(clf["best_estimator"]).__name__, clf["best_estimator"]))
        means.append(clf["mean"])
        i = i+1

    return clfs, means


def voting_predict(clfs, mean, train_x, train_y, test_x, test_id):

    clf = VotingClassifier(estimators=clfs, weights = mean, voting = 'soft')
    clf.fit(train_x, train_y.values.ravel())
    pred = clf.predict(test_x)
    print(test_id.shape, pred.shape)
    prediction_df = pd.DataFrame({'PetID': test_id.values.ravel(),
                                  'AdoptionSpeed': pred})

    return prediction_df



if __name__ == "__main__":
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1500)
    df = pd.DataFrame(np.random.randn(5, 3), columns=['perf', 'B', 'C'])

    #predict(df)
    train, test = read_data()

    train_x, train_y, test_x, test_id = prepare_data(train, test)
    #rs = randomsearchpipeline(train_x.drop(["RescuerID"], axis=1), train_y)
    train_x.drop(["RescuerID"], axis=1, inplace=True)
    test_x.drop(["RescuerID"], axis=1, inplace=True)
    #print(train_x)
    #print(rs)
    #sys.exit()
    #print(train_x)
    #print(train_y)
    #print(sp.stats.randint(1, 6).value)

    clf, mean = iterate_by_randomsearch(train_x, train_y.values.ravel())
    print(clf)
    print(mean)
    #for c in clf[""]:
        #print(c)
    #test_bayes(train_x, train_y)
    voting_predict(clf, mean, train_x, train_y, test_x, test_id)
    #print(pred)



# to do
# different algorithms rather than the versions of same
# input tfidf
# predict differently for cat and dogs
