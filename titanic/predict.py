# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Albert Thomas <albert.thomas@telecom-paristech.fr>
# License: BSD 3 clause

import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

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

def iterate_clf(train_x, train_y):

    classifiers = [
        AdaBoostClassifier(n_estimators=50),
        BaggingClassifier(n_estimators=50),
        ExtraTreesClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators=50),
        IsolationForest(n_estimators=50, contamination=0.005, behaviour='new' ),
        RandomForestClassifier(n_estimators=50),
        #VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard'),
        GaussianProcessClassifier(),
        LogisticRegression(solver='liblinear'),
        LogisticRegressionCV(solver='liblinear',cv=5),
        PassiveAggressiveClassifier(max_iter=1000 , tol= 0.001),
        RidgeClassifier(max_iter=1000 , tol= 0.001),
        RidgeClassifierCV(cv=5),
        SGDClassifier(max_iter=1000 , tol= 0.001),
        BernoulliNB(),
        MultinomialNB(),
        KNeighborsClassifier(),
        #RadiusNeighborsClassifier(),
        NearestCentroid(),
        MLPClassifier(max_iter=1000),
        DecisionTreeClassifier(),
        ExtraTreeClassifier(),
        LinearSVC(max_iter=200),
        NuSVC(gamma= 'RBF',max_iter=200),
        SVC(gamma= 'RBF',max_iter=200),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()
    ]

    for clf in classifiers:
        start = datetime.now()
        scores = cross_val_score(clf, train_x, train_y, cv=5,scoring="accuracy")
        end = datetime.now()
        print(start, end, end-start,  type(clf).__name__, scores.mean(), scores.std())



if __name__ == "__main__":
    train_x, train_y, test, gender_sub = prepare()
    iterate_clf(train_x,train_y)