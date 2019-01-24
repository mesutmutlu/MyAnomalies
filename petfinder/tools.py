from petfinder.get_explore import read_data
from sklearn.preprocessing import LabelEncoder
from enum import Enum
from petfinder.get_explore import Columns, Paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from nltk.corpus import stopwords
import nltk
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import datetime
from collections import Counter
import os
def create_dict(file, idx, col):
    df = pd.read_csv(file)
    df.set_index(idx, inplace = True)
    dct = df[col].to_dict()
    dct[0] = "Not Set"
    return dct


def tfidf_2(train, n_comp, out_cols):
    print("starting tfidf")
    stop_words = set(stopwords.words('english'))
    tfv = TfidfVectorizer(min_df=2, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words=stop_words)

    # Fit TFIDF
    tfv.fit(list(train))
    X = tfv.transform(train)
    print("length", X.shape)
    svd = TruncatedSVD(n_components=n_comp)
    svd.fit(X)
    print(svd.explained_variance_ratio_.sum())
    #print(svd.explained_variance_ratio_)
    X = svd.transform(X)
    df_svd = pd.DataFrame(X, columns=out_cols)

    return df_svd


def label_encoder(arr, cols):
    enc = LabelEncoder()
    for col in cols:
        enc_labels = enc.fit_transform(arr[col])

        arr[col] = enc_labels
    return arr


def scale_num_var(train, test):
    for col in Columns.ind_cont_columns.value:
        train[col] = (train[col]-train[col].mean())/train[col].std()
        test[col] = (test[col]-train[col].mean())/train[col].std()
    return train, test


def fill_na(arr, cols, val):
    for col in cols:
        arr[col].fillna(val, inplace=True)
    return arr

def detect_outliers(train, rc):
    print("detecting outliers")
    if rc == 1 or not(os.path.isfile("outliers.csv")):

        f_train = train.drop(["PetID"], axis=1)
        id = train["PetID"]
        n_samples = len(f_train)
        outliers_fraction = 0.10
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers

        # define outlier/anomaly detection methods to be compared
        anomaly_algorithms = [
            # ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
            ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                              gamma=0.1)),
            ("Isolation Forest", IsolationForest(behaviour='new',
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

        df_pred["outlier"] = df_pred.apply(lambda x: Counter([x['One-Class SVM'], x['Isolation Forest'], x["Local Outlier Factor"]]).most_common(1)[0][0], axis=1)

        prediction_df = pd.concat([id,df_pred], axis=1)

        # create submission file print(prediction_df)
        prediction_df.to_csv("outliers.csv")
    prediction_df = pd.read_csv("outliers.csv")
    return prediction_df["outlier"]

if __name__ == "__main__":

    nltk.download()