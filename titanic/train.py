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
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from time import time
from numpy.random import uniform as np_uniform

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

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
        (AdaBoostClassifier(), {"n_estimators": sp_randint(25, 100)}),
        # BaggingClassifier(n_estimators=50),
        # ExtraTreesClassifier(n_estimators=50),
        # GradientBoostingClassifier(n_estimators=50),
        # IsolationForest(n_estimators=50, contamination=0.005, behaviour='new' ),
        (RandomForestClassifier(), {"n_estimators": sp_randint(25, 100),
                                    "max_depth": [3, None],
                                    "max_features": sp_randint(1, 6),
                                    "min_samples_split": sp_randint(2, 11),
                                    "bootstrap": [True, False],
                                    "criterion": ["gini", "entropy"]})]

    for clf in classifiers:

        random_search = RandomizedSearchCV(clf[0], param_distributions=clf[1],
                                           n_iter=20, cv=5)
        start = time()
        random_search.fit(train_x, train_y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), 20))
        report(random_search.cv_results_)

def iterate_by_gridsearch(train_x, train_y):
    classifiers = [
        (AdaBoostClassifier(), {"n_estimators": sp_randint(25, 100)}),
        # BaggingClassifier(n_estimators=50),
        # ExtraTreesClassifier(n_estimators=50),
        # GradientBoostingClassifier(n_estimators=50),
        # IsolationForest(n_estimators=50, contamination=0.005, behaviour='new' ),
        (RandomForestClassifier(), {"n_estimators": sp_randint(25, 100),
                                    "max_depth": [3, None],
                                    "max_features": sp_randint(1, 6),
                                    "min_samples_split": sp_randint(2, 11),
                                    "bootstrap": [True, False],
                                    "criterion": ["gini", "entropy"]})]

    for clf in classifiers:

        random_search = GridSearchCV(clf[0], param_grid=clf[1], cv=5)
        start = time()
        random_search.fit(train_x, train_y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), 20))
        report(random_search.cv_results_)



def iterate_clf(train_x, train_y):

    classifiers = [
        (AdaBoostClassifier(),{"n_estimators":sp_randint(25,100),
                               "max_depth": [3, None]}),
        # BaggingClassifier(n_estimators=50),
        # ExtraTreesClassifier(n_estimators=50),
        GradientBoostingClassifier(),{"n_estimators":sp_randint(25,100),
                                      "loss":["deviance", "exponential"],
                                      "subsample": np_uniform(0.3, 1),
                                      "max_features": sp_randint(1, 6),
                                      "min_samples_split": sp_randint(2, 11),
                                      "bootstrap": [True, False],
                                      "criterion": ["friedman_mse", "mse", "mae"],
                                      "max_depth": [3, None]},
        # IsolationForest(n_estimators=50, contamination=0.005, behaviour='new' ),
        (RandomForestClassifier(),{"n_estimators":sp_randint(25,100),
                                    "max_depth": [3, None],
                                    "max_features": sp_randint(1, 6),
                                    "min_samples_split": sp_randint(2, 11),
                                    "bootstrap": [True, False],
                                    "criterion": ["gini", "entropy"]}),
        # #VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard'),
        # GaussianProcessClassifier(),
        # LogisticRegression(solver='liblinear'),
        # LogisticRegressionCV(solver='liblinear',cv=5),
        # PassiveAggressiveClassifier(max_iter=1000 , tol= 0.001),
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

    train_x, train_y, test, gender_sub = prepare()
    iterate_by_gridsearch(train_x, train_y)
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