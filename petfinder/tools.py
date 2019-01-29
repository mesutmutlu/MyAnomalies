from petfinder.get_explore import read_data
from sklearn.preprocessing import LabelEncoder
from enum import Enum
from petfinder.get_explore import Columns, Paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from nltk.corpus import stopwords
import sys
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import datetime
from collections import Counter
import os
import featuretools as ft
import seaborn as sns
from time import time
from sklearn import manifold
import matplotlib.pyplot as plt

def plog(msg):
    print(datetime.datetime.now(), msg)

def create_dict(file, idx, col):
    df = pd.read_csv(file)
    df.set_index(idx, inplace = True)
    dct = df[col].to_dict()
    dct[0] = "Not Set"
    return dct


def tfidf_2(train, n_comp, out_cols):
    print("starting tfidf")
    #train.replace("", "none")
    stop_words = set(stopwords.words('english'))
    tfv = TfidfVectorizer(min_df=3, max_features=10000,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = 'english')

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
    if rc == 1 or not(os.path.isfile(Paths.base.value + "outliers.csv")):

        f_train = train
        n_samples = len(f_train)
        outliers_fraction = 0.10
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers

        # define outlier/anomaly detection methods to be compared

        anomaly_algorithms = [
            # ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
            # ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
            #                                   gamma=0.1)),
            ("Isolation Forest", IsolationForest(#behaviour='new',
                                                 contamination=outliers_fraction,
                                                 random_state=42)),
            # ("Local Outlier Factor", LocalOutlierFactor(
            #     n_neighbors=35, contamination=outliers_fraction))
        ]

        # Define datasets
        #print(f_train.head())
        df_pred = pd.DataFrame(columns=["One-Class SVM","Isolation Forest","Local Outlier Factor"])
        if 1 == 1 :
            for name, algorithm in anomaly_algorithms:
                #t0 = time.time()
                # algorithm.fit(f_train)
                plog(name + " started")
                #t1 = time.time()

                # fit the data and tag outliers
                if name == "Local Outlier Factor":
                    y_pred = algorithm.fit_predict(f_train)
                else:
                    y_pred = algorithm.fit(f_train).predict(f_train)
                plog(name + " ended")
                #print(name, y_pred)
                df_pred[name] = y_pred
                #print(df_pred)


        df_pred["outlier"] = df_pred.apply(lambda x: Counter([x['One-Class SVM'], x['Isolation Forest'], x["Local Outlier Factor"]]).most_common(1)[0][0], axis=1)

        #prediction_df = pd.concat([id,df_pred], axis=1)

        # create submission file print(prediction_df)
        df_pred.to_csv(Paths.base.value + "outliers.csv")
        df_pred = pd.read_csv(Paths.base.value + "outliers.csv")
    return df_pred["outlier"]

def auto_features(df, cols, entities):

    df_c = df[cols]
    es = ft.EntitySet(id='petfinder')
    es.entity_from_dataframe(entity_id="Pets", dataframe=df_c, index="PetID")
    ignored_variable =  {}
    ignored_variable.update({'Pets': entities})
    for e in entities:
        print(e)

        es.normalize_entity(base_entity_id='Pets', new_entity_id=e, index=e)
        feature_matrix, feature_names = ft.dfs(entityset=es,
                                               target_entity=e,
                                               max_depth=2,
                                               verbose=1,
                                               #n_jobs=3,
                                               ignore_variables=ignored_variable)
        fm = feature_matrix.add_prefix(e+"_")
        df = df.set_index(e).join(fm).reset_index()

    return df

def tsne(x_train, y_train):
    perplexities = [5, 30, 50, 100]
    # (fig, subplots) = plt.subplots(1, 5, figsize=(15, 8))
    print(len(x_train), len(y_train))
    #    ax = subplots[0][0]
    print("tsne started")

    for i, perplexity in enumerate(perplexities):
        # ax = subplots[0][i + 1]

        t0 = time()
        print(datetime.datetime.now())
        tsne = manifold.TSNE(n_components=2, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(x_train)
        print(datetime.datetime.now())
        t1 = time()
        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        print(len(x_train), len(Y))
        print(Y)
        data = pd.concat([pd.DataFrame(data=Y, columns=["tsne1", "tsne2"]), y_train], axis=1)
        print(data)
        # ax.set_title("Perplexity=%d" % perplexity)
        sns.scatterplot(x="tsne1", y="tsne2", hue="AdoptionSpeed", data=data, legend="full")
        plt.show()

        # ax.axis('tight')

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    c = Columns.ind_num_cat_columns.value.copy()
    print(c)
    c.remove("RescuerType")
    c.remove("RescuerID")
    c2 = Columns.ind_cont_columns.value.copy()
    c2.remove("NameLength")
    c2.remove("DescLength")
    train_x= train[c2 + c]
    train_x.fillna(0, inplace=True)
    train_y = train["AdoptionSpeed"]
    print(train_x.describe(include="all"))
    tsne(train_x, train_y)