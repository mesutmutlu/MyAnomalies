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
        # thsis is for anomaly detection (IsolationForest(),{"n_estimators":50,
        #                     "contamination":np_uniform(0., 0.5),
        #                     "behaviour":["old", "new"],
        #                     "bootstrap": [True, False],
        #                     "max_features": sp.stats.randint(1, 7),
        #                     "min_samples_split": sp.stats.randint(2, 11)}),
        # this is for outlier detection (RadiusNeighborsClassifier(), {"radius": sp.stats.uniform(0.5, 5),
        #                                "algorithm": ["ball_tree", "kd_tree", "brute"],
        #                                "leaf_size": sp.stats.randint(20, 100),
        #                                "p": [1, 2]})
        (AdaBoostClassifier(), {"n_estimators": sp.stats.randint(25, 100)}),
        (BaggingClassifier(),{"n_estimators": sp.stats.randint(25, 100),
                                             "max_features": sp.stats.randint(1, 7),
                                             "bootstrap": [True, False],
                                             "bootstrap_features": [True, False],
                                                }),
        (ExtraTreesClassifier(), {"n_estimators": sp.stats.randint(25, 100),
                                    "max_depth": [3, None],
                                    "max_features": sp.stats.randint(1, 7),
                                    "min_samples_split": sp.stats.randint(2, 11),
                                    "bootstrap": [True, False],
                                    "criterion": ["gini", "entropy"]}),
        (GradientBoostingClassifier(), {"n_estimators": sp.stats.randint(25, 100),
                                       "loss": ["deviance", "exponential"],
                                       "max_features": sp.stats.randint(1, 7),
                                       "min_samples_split": sp.stats.randint(2, 11),
                                       "criterion": ["friedman_mse", "mse", "mae"],
                                       "max_depth": [3, None]}),
        (RandomForestClassifier(), {"n_estimators": sp.stats.randint(25, 100),
                                    "max_depth": [3, None],
                                    "max_features": sp.stats.randint(1, 7),
                                    "min_samples_split": sp.stats.randint(2, 11),
                                    "bootstrap": [True, False],
                                    "criterion": ["gini", "entropy"]}),
        #(GaussianProcessClassifier(), {}),
        # (LogisticRegression(), { "max_iter":sp.stats.randint(0,500),
        #                        "solver":["sag", "saga"],
        #                          "multi_class":["auto"]}),
        (PassiveAggressiveClassifier(), {"max_iter":sp.stats.randint(0, 1230),
                                         "tol": sp.stats.uniform(0.0001, 0.05)}),
        (RidgeClassifier(), {"max_iter":sp.stats.randint(0, 2000),
                             "tol": sp.stats.uniform(0.0001, 0.05),
                             "solver":["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}),
        (SGDClassifier(), {"max_iter":sp.stats.randint(0, 2000),
                              "tol": sp.stats.uniform(0.0001, 0.05),
                           "loss":["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                            "penalty":["none", "l2", "l1", "elasticnet"]}),
        #(BernoulliNB(), {}),
        #(MultinomialNB(), {}),
        #(GaussianNB(), {}),
        #(ComplementNB(), {}),
        (KNeighborsClassifier(), {"n_neighbors":sp.stats.randint(1, 50),
                                  "algorithm":["ball_tree", "kd_tree", "brute"],
                                  "leaf_size":sp.stats.randint(20,100),
                                  "p":[1,2]}),
        (NearestCentroid(),{}),
        # (MLPClassifier(), {"hidden_layer_sizes":(random.randint(10,1000),),
        #                   "activation":["identity", "logistic", "tanh", "relu"],
        #                   "solver": ["lbfgs", "sgd", "adam"],
        #                    "alpha":sp.stats.uniform(0.00001, 0.001),
        #                    "learning_rate":["constant", "invscaling", "adaptive"],
        #                    "max_iter": sp.stats.randint(0, 2000),
        #                    "tol": sp.stats.uniform(0.0001, 0.05)}),
        (DecisionTreeClassifier(), {          "max_depth": [3, None],
                                              "max_features": sp.stats.randint(1, 7),
                                              "min_samples_split": sp.stats.randint(2, 11),
                                              "criterion": ["gini", "entropy"]}),
        # (LinearSVC(), {"penalty":["l2"],
        #         #                "tol":sp.stats.uniform(1e-5, 1e-3),
        #         #                "C":sp.stats.uniform(0.1, 5),
        #         #                "max_iter":sp.stats.randint(0, 2000)}),
        # (NuSVC(),{"gamma":sp.stats.uniform(1e-4, 1e-2),
        #           "kernel":["linear", "poly", "rbf", "sigmoid"],
        #           "tol":sp.stats.uniform(1e-4, 1e-2),
        #           }),
        # (SVC(), {"gamma":sp.stats.uniform(1e-4, 1e-2),
        #                   "kernel":["linear", "poly", "rbf", "sigmoid"],
        #                   "tol":sp.stats.uniform(1e-4, 1e-2),}),
        # (LinearDiscriminantAnalysis(),{"solver":["svd","lsqr", "eigen"],
        #                                   "n_components":random.randint(2,4),
        #                                    "tol":sp.stats.uniform(1e-5, 1e-2)
        # }),
        (QuadraticDiscriminantAnalysis(), {"tol":sp.stats.uniform(1e-5, 1e-2)})


    ]

    df = pd.DataFrame(columns=['alg', 'best_estimator', 'perf', 'est','rank','mean','std', 'parameters'])
    for clf in classifiers:
        print(type(clf[0]).__name__, "started at", datetime.now())
        n_iter=20
        kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
        random_search = RandomizedSearchCV(clf[0], param_distributions=clf[1],
                                           n_iter=n_iter, cv=5, scoring=kappa_scorer)
        start = time()
        random_search.fit(train_x, train_y)
        #print("%s RandomizedSearchCV took %.2f seconds for %d candidates"
        #      " parameter settings." % (type(clf[0]).__name__,(time() - start), n_iter))

        df = report(df, type(clf[0]).__name__, random_search.best_estimator_ ,  time() - start, n_iter, random_search.cv_results_)
    print(df.sort_values(by="mean", ascending=False))
    best = df['mean'].idxmax()

    est = df.loc[best, 'best_estimator']
    print(est)
    return est


class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass


def randomsearchpipeline(train_x, train_y):
    classifiers = [
        # thsis is for anomaly detection (IsolationForest(),{"n_estimators":50,
        #                     "contamination":np_uniform(0., 0.5),
        #                     "behaviour":["old", "new"],
        #                     "bootstrap": [True, False],
        #                     "max_features": sp.stats.randint(1, 7),
        #                     "min_samples_split": sp.stats.randint(2, 11)}),
        # this is for outlier detection (RadiusNeighborsClassifier(), {"radius": sp.stats.uniform(0.5, 5),
        #                                "algorithm": ["ball_tree", "kd_tree", "brute"],
        #                                "leaf_size": sp.stats.randint(20, 100),
        #                                "p": [1, 2]})
        {"clf" : [AdaBoostClassifier()], "n_estimators": sp.stats.randint(25, 100)},
        {"clf" : [BaggingClassifier()],"n_estimators": sp.stats.randint(25, 100),
                                              "max_features": sp.stats.randint(1, 7),
                                              "bootstrap": [True, False],
                                              "bootstrap_features": [True, False],
                                                 },
        # (ExtraTreesClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                             "max_depth": [3, None],
        #                             "max_features": sp.stats.randint(1, 7),
        #                             "min_samples_split": sp.stats.randint(2, 11),
        #                             "bootstrap": [True, False],
        #                             "criterion": ["gini", "entropy"]}),
        # # (GradientBoostingClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        # #                                "loss": ["deviance", "exponential"],
        # #                                "max_features": sp.stats.randint(1, 7),
        # #                                "min_samples_split": sp.stats.randint(2, 11),
        # #                                "criterion": ["friedman_mse", "mse", "mae"],
        # #                                "max_depth": [3, None]}),
        # (RandomForestClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                             "max_depth": [3, None],
        #                             "max_features": sp.stats.randint(1, 7),
        #                             "min_samples_split": sp.stats.randint(2, 11),
        #                             "bootstrap": [True, False],
        #                             "criterion": ["gini", "entropy"]}),
        # #(GaussianProcessClassifier(), {}),
        # (LogisticRegression(), { "max_iter":sp.stats.randint(0,100),
        #                        "solver":["lbfgs", "sag", "saga"]}),
        # (PassiveAggressiveClassifier(), {"max_iter":sp.stats.randint(0, 1230),
        #                                  "tol": sp.stats.uniform(0.0001, 0.05)}),
        # (RidgeClassifier(), {"max_iter":sp.stats.randint(0, 2000),
        #                      "tol": sp.stats.uniform(0.0001, 0.05),
        #                      "solver":["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}),
        # (SGDClassifier(), {"max_iter":sp.stats.randint(0, 2000),
        #                       "tol": sp.stats.uniform(0.0001, 0.05),
        #                    "loss":["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        #                     "penalty":["none", "l2", "l1", "elasticnet"]}),
        # (BernoulliNB(), {}),
        # (MultinomialNB(), {}),
        # (GaussianNB(), {}),
        # (ComplementNB(), {}),
        # (KNeighborsClassifier(), {"n_neighbors":sp.stats.randint(1, 50),
        #                           "algorithm":["ball_tree", "kd_tree", "brute"],
        #                           "leaf_size":sp.stats.randint(20,100),
        #                           "p":[1,2]}),
        # (NearestCentroid(),{}),
        # (MLPClassifier(), {"hidden_layer_sizes":(random.randint(10,1000),),
        #                   "activation":["identity", "logistic", "tanh", "relu"],
        #                   "solver": ["lbfgs", "sgd", "adam"],
        #                    "alpha":sp.stats.uniform(0.00001, 0.001),
        #                    "learning_rate":["constant", "invscaling", "adaptive"],
        #                    "max_iter": sp.stats.randint(0, 2000),
        #                    "tol": sp.stats.uniform(0.0001, 0.05)}),
        # (DecisionTreeClassifier(), {          "max_depth": [3, None],
        #                                       "max_features": sp.stats.randint(1, 7),
        #                                       "min_samples_split": sp.stats.randint(2, 11),
        #                                       "criterion": ["gini", "entropy"]}),
        # (LinearSVC(), {"penalty":["l2"],
        #                "tol":sp.stats.uniform(1e-5, 1e-3),
        #                "C":sp.stats.uniform(0.1, 5),
        #                "max_iter":sp.stats.randint(0, 2000)}),
        # (NuSVC(),{"gamma":sp.stats.uniform(1e-4, 1e-2),
        #           "kernel":["linear", "poly", "rbf", "sigmoid"],
        #           "tol":sp.stats.uniform(1e-4, 1e-2),
        #           }),
        # (SVC(), {"gamma":sp.stats.uniform(1e-4, 1e-2),
        #                   "kernel":["linear", "poly", "rbf", "sigmoid"],
        #                   "tol":sp.stats.uniform(1e-4, 1e-2),}),
        # (LinearDiscriminantAnalysis(),{"solver":["svd","lsqr", "eigen"],
        #                                   "n_components":random.randint(2,4),
        #                                    "tol":sp.stats.uniform(1e-5, 1e-2)
        # }),
        # (QuadraticDiscriminantAnalysis(), {"tol":sp.stats.uniform(1e-5, 1e-2)})


    ]

    pipe = Pipeline([("clf", DummyEstimator())])
    random_search = RandomizedSearchCV(pipe, classifiers,
                                           n_iter=2, cv=5)
    random_search.fit(train_x.values, train_y.values.ravel())
    return random_search

def predict(clf, train_x, train_y, test_x, test_id):
    clf.fit(train_x, train_y.values.ravel())
    pred = clf.predict(test_x)
    print(test_id.shape, pred.shape)
    prediction_df = pd.DataFrame({'PetID': test_id.values.ravel(),
                                  'AdoptionSpeed': pred})

    print(prediction_df)



if __name__ == "__main__":
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1500)
    df = pd.DataFrame(np.random.randn(5, 3), columns=['perf', 'B', 'C'])

    #predict(df)

    train_x, train_y, test_x, test_id = prepare_data()
    #rs = randomsearchpipeline(train_x.drop(["RescuerID"], axis=1), train_y)
    train_x.drop(["RescuerID"], axis=1, inplace=True)
    test_x.drop(["RescuerID"], axis=1, inplace=True)
    #print(rs)
    #sys.exit()
    #print(train_x)
    #print(train_y)
    #print(sp.stats.randint(1, 6).value)
    clf = iterate_by_randomsearch(train_x, train_y.values.ravel())
    #test_bayes(train_x, train_y)
    predict(clf, train_x, train_y, test_x, test_id)
    #print(pred)

