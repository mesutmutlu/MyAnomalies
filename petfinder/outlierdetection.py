import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from petfinder.get_explore import read_data
from petfinder.preprocessing import prepare_data
import pandas as pd
import sys
from sklearn.ensemble import VotingClassifier
import datetime
from collections import Counter

print(__doc__)


def MahalanobisDist(data):
    covariance_xyz = np.cov(data) # calculate the covarince matrix
    inv_covariance_xyz = np.linalg.inv(covariance_xyz) #take the inverse of the covarince matrix
    xyz_mean = np.mean(data[0]),np.mean(data[1]),np.mean(data[2])
    x_diff = np.array([x_i - xyz_mean[0] for x_i in x]) # take the diffrence between the mean of X variable the sample
    y_diff = np.array([y_i - xyz_mean[1] for y_i in y]) # take the diffrence between the mean of Y variable the sample
    z_diff = np.array([z_i - xyz_mean[2] for z_i in z]) # take the diffrence between the mean of Z variable the sample
    diff_xyz = np.transpose([x_diff, y_diff, z_diff])

    md = []
    for i in range(len(diff_xyz)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xyz[i]),inv_covariance_xyz),diff_xyz[i]))) #calculate the Mahalanobis Distance for each data sample
    return md

def MD_detectOutliers(data):
    MD = MahalanobisDist(data)
    threshold = np.mean(MD) * 1.5 # adjust 1.5 accordingly
    outliers = []
    for i in range(len(MD)):
        if MD[i] > threshold:
            outliers.append(i) # index of the outlier
    df_outliers = pd.DataFrame(data=outliers)
    df_outliers.to_csv("mala.csv", delimiter=",")

def detect_outliers(f_train, id):
    n_samples = len(f_train)
    outliers_fraction = 0.10
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # define outlier/anomaly detection methods to be compared
    anomaly_algorithms = [
        # ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                          gamma=0.1)),
        ("Isolation Forest", IsolationForest(#behaviour='new',
                                             contamination=outliers_fraction,
                                             random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(
            n_neighbors=35, contamination=outliers_fraction))]

    # Define datasets
    #print(f_train.head())
    df_pred = pd.DataFrame(columns=["One-Class SVM","Isolation Forest","Local Outlier Factor"])
    if 1 == 1 :
        for name, algorithm in anomaly_algorithms:
            #t0 = time.time()
            # algorithm.fit(f_train)
            print(name, datetime.datetime.now())
            #t1 = time.time()

            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(f_train)
            else:
                y_pred = algorithm.fit(f_train).predict(f_train)

            #print(name, y_pred)
            df_pred[name] = y_pred
            #print(df_pred)
    if 1 == 0:
        clf = VotingClassifier(estimators=anomaly_algorithms, voting='soft')
        y_pred = clf.fit(f_train, y=None).predict(f_train)
        # print(test_id.shape, pred.shape)
    for index, row in df_pred.iterrows():

        df_pred["final_class"] = Counter(row[["One-Class SVM","Isolation Forest","Local Outlier Factor"]].values.ravel()).most_common(1)[0][0]

    prediction_df = pd.concat([id,df_pred.values], axis=1)

    # create submission file print(prediction_df)
    return prediction_df
if __name__ == "__main__":
    L = [1, 2, 45, 55, 5, 4, 4, 4, 4, 4, 4, 5456, 56, 6, 7, 67]
    print(Counter(L).most_common(1)[0][0])
    most_common, num_most_common = Counter(L).most_common(1)[0]  # 4, 6 times

    train, test = read_data()

    x_train, y_train, x_test, id_test = prepare_data(train, test)

    f_train = pd.concat([x_train, y_train], axis=1)

    #print(f_train.values)
    #print(f_train.columns.values)
    pred = detect_outliers(f_train.drop(["PetID"], axis=1), f_train["PetID"])
    print(pred)