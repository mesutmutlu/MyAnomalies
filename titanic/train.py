# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Albert Thomas <albert.thomas@telecom-paristech.fr>
# License: BSD 3 clause

import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import sys

from titanic.explore import prepare
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
from skopt import gp_minimize
from skopt import WeightedBayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from time import time
from numpy.random import random_integers #integers
from numpy.random import random_sample #floats
import pandas as pd
import scipy as sp

def report(df, alg, perf, est, results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            #print("Model with rank: {0}".format(i))
            #print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            #      results['mean_test_score'][candidate],
            #      results['std_test_score'][candidate]))
            #print("Parameters: {0}".format(results['params'][candidate]))
            #print("")
            df.loc[len(df)] = [alg, perf, est, format(i), results['mean_test_score'][candidate],results['std_test_score'][candidate],results['params'][candidate]]
    return df


def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)


def test_bayes(X_train, y_train):
    opt = WeightedBayesSearchCV(SVC(), { 'C': Real(1e-6, 1e+6, prior='log-uniform'),
                                         'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                                         'degree': Integer(1,8),
                                         'kernel': Categorical(['linear', 'poly', 'rbf']), }, n_iter=32, cv=10, scoring="accuracy")
    opt.fit(X_train, y_train)
    print("val. score: %s" % opt.best_score_)

def iterate_by_randomsearch(train_x, train_y):
    classifiers = [
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
        #thsis is for anomaly detection (IsolationForest(),{"n_estimators":50,
        #                     "contamination":np_uniform(0., 0.5),
        #                     "behaviour":["old", "new"],
        #                     "bootstrap": [True, False],
        #                     "max_features": sp.stats.randint(1, 7),
        #                     "min_samples_split": sp.stats.randint(2, 11)}),
        (RandomForestClassifier(), {"n_estimators": sp.stats.randint(25, 100),
                                    "max_depth": [3, None],
                                    "max_features": sp.stats.randint(1, 7),
                                    "min_samples_split": sp.stats.randint(2, 11),
                                    "bootstrap": [True, False],
                                    "criterion": ["gini", "entropy"]}),
        (GaussianProcessClassifier(), {"tol": sp.stats.uniform(0.0001, 0.05)}),
        (LogisticRegression(), { "max_iter":sp.stats.randint(0,100),
                               "solver":["lbfgs", "sag", "saga"]}),
        (PassiveAggressiveClassifier(), {"max_iter":sp.stats.randint(0, 1230),
                                         "tol": sp.stats.uniform(0.0001, 0.05)})
    ]
    df = pd.DataFrame(columns=['alg', 'perf', 'est','rank','mean','std', 'parameters'])
    for clf in classifiers:
        #print(clf)
        n_iter=10
        random_search = RandomizedSearchCV(clf[0], param_distributions=clf[1],
                                           n_iter=n_iter, cv=5)
        start = time()
        random_search.fit(train_x, train_y)
        #print("%s RandomizedSearchCV took %.2f seconds for %d candidates"
        #      " parameter settings." % (type(clf[0]).__name__,(time() - start), n_iter))

        df = report(df, type(clf[0]).__name__, time() - start, n_iter, random_search.cv_results_)
        print(df)

def iterate_by_gridsearch(train_x, train_y):
    estimators = random_integers(25, 100, size=5)
    print(estimators)
    max_features = random_integers(1, 7, size=6)
    min_samples_split = random_integers(2, 11, size=5)
    subsample = random_sample((5,))
    classifiers = [
        (AdaBoostClassifier(), {"n_estimators": estimators}),
        # BaggingClassifier(n_estimators=50),
        # ExtraTreesClassifier(n_estimators=50),
        (GradientBoostingClassifier(), {"n_estimators": estimators,
                                        "loss": ["deviance", "exponential"],
                                        "max_features": max_features,
                                        "min_samples_split": min_samples_split,
                                        "criterion": ["friedman_mse", "mse", "mae"],
                                        "max_depth": [3, None]}),
        # IsolationForest(n_estimators=50, contamination=0.005, behaviour='new' ),
        (RandomForestClassifier(), {"n_estimators": estimators,
                                    "max_depth": [3, None],
                                    "max_features": max_features,
                                    "min_samples_split": min_samples_split,
                                    "bootstrap": [True, False],
                                    "criterion": ["gini", "entropy"]})]

    for clf in classifiers:
        print("----------", time(), clf[0], "-----------")
        random_search = GridSearchCV(clf[0], param_grid=clf[1], cv=5)
        start = time()
        random_search.fit(train_x, train_y)
        print("GridSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), 20))
        report(random_search.cv_results_)

def PassiveAggressiveClassifier_test(train_x, train_y):
    estimators = random_integers(25, 100, size=5)
    print(estimators)
    max_features = random_integers(1, 7, size=6)
    min_samples_split = random_integers(2, 11, size=5)
    subsample = random_sample((5,))
    params = {"max_iter": sp.stats.randint(0, 1230),
            "tol": sp.stats.uniform(0.001, 0.5)}

    random_search = RandomizedSearchCV(PassiveAggressiveClassifier(), param_distributions=params, cv=10, n_iter=5)
    start = time()
    random_search.fit(train_x, train_y)
    print("GridSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), 20))
    report(random_search.cv_results_)

def gradient_test(train_x, train_y):
    estimators = random_integers(25, 100, size=5)
    print(estimators)
    max_features = random_integers(1, 7, size=6)
    min_samples_split = random_integers(2, 11, size=5)
    subsample = random_sample((5,))
    params = {"n_estimators": sp.stats.randint(25, 100),
                                       "loss": ["deviance", "exponential"],
                                       "max_features": sp.stats.randint(1, 7),
                                       "min_samples_split": sp.stats.randint(2, 11),
                                       "criterion": ["friedman_mse", "mse", "mae"],
                                       "max_depth": [3, None]}

    random_search = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=params, cv=10, n_iter=5)
    start = time()
    random_search.fit(train_x, train_y)
    print("GridSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), 20))
    report(random_search.cv_results_)


def iterate_clf(train_x, train_y):
    classifiers = [
        #AdaBoostClassifier(n_estimators=100),
        # BaggingClassifier(n_estimators=50),
        # ExtraTreesClassifier(n_estimators=50),
        #GradientBoostingClassifier(n_estimators=100),
        # IsolationForest(n_estimators=50, contamination=0.005, behaviour='new' ),
        #RandomForestClassifier(n_estimators=100),
        # #VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard'),
        # GaussianProcessClassifier(),
        # LogisticRegression(solver='liblinear'),
        # LogisticRegressionCV(solver='liblinear',cv=5),
        #PassiveAggressiveClassifier(max_iter=1000 , tol= 0.001),
        # RidgeClassifier(max_iter=1000 , tol= 0.001),
        # RidgeClassifierCV(cv=5),
        # SGDClassifier(max_iter=1000 , tol= 0.001),
        # BernoulliNB(),
        # MultinomialNB(),
        # KNeighborsClassifier(),
        # #RadiusNeighborsClassifier(),
        # NearestCentroid(),
        # MLPClassifier(max_iter=1000),
        # DecisionTreeClassifier(),
        # ExtraTreeClassifier(),
        # LinearSVC(),
        # NuSVC(gamma=0.001),
        # SVC(gamma=0.001),
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis()
    ]

    for clf in classifiers:
        start = datetime.now()
        scores = cross_val_score(clf, train_x, train_y, cv=10,scoring="accuracy")
        end = datetime.now()
        print(start, end, end-start,  type(clf).__name__, scores.mean(), scores.std())



if __name__ == "__main__":
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    train_x, train_y, test, gender_sub = prepare()
    #print(sp.stats.randint(1, 6).value)
    iterate_by_randomsearch(train_x, train_y)
    #test_bayes(train_x, train_y)
    sys.exit()
    train_x.drop(["Fare","Age","Pclass"], axis=1, inplace=True)
    print(train_x.head())
    print(train_x.Family.unique())
    #sys.exit()
    print(train_x.head().values)
    #print(train_y.head())
    scaler = MinMaxScaler()
    iterate_clf(scaler.fit_transform(train_x.values),train_y.values)
    pca = PCA(n_components=5)
    pca.fit(train_x.values)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)