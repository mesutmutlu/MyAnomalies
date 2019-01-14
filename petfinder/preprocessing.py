import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2 as sk_chi, SelectKBest
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import entropy
import numpy as np
from collections import Counter
import math
from petfinder.get_explore import read_data
from sklearn.preprocessing import LabelEncoder
from enum import Enum
from petfinder.get_explore import Columns, Paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def fill_na(arr, cols, val):
    for col in cols:
        arr[col].fillna(val, inplace=True)
    return arr

def tfidf(train , test, n_svdcomp, n_iann_svdcomp):
    train_desc = train.Description
    test_desc = test.Description

    tfv = TfidfVectorizer(min_df=2, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          )

    # Fit TFIDF
    tfv.fit(list(train_desc))
    X = tfv.transform(train_desc)
    X_test = tfv.transform(test_desc)

    svd = TruncatedSVD(n_components=120)
    svd.fit(X)
    #print(svd.explained_variance_ratio_.sum())
    #print(svd.explained_variance_ratio_)
    X = svd.transform(X)
    train_desc = pd.DataFrame(X, columns=Columns.desc_cols.value)
    X_test = svd.transform(X_test)
    test_desc = pd.DataFrame(X_test, columns=Columns.desc_cols.value)

    train_desc = train.Description
    test_desc = test.Description

    tfv = TfidfVectorizer(min_df=2, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          )

    # Fit TFIDF
    tfv.fit(list(train_desc))
    X = tfv.transform(train_desc)
    X_test = tfv.transform(test_desc)

    svd = TruncatedSVD(n_components=120)
    svd.fit(X)
    # print(svd.explained_variance_ratio_.sum())
    # print(svd.explained_variance_ratio_)
    X = svd.transform(X)
    train_desc = pd.DataFrame(X, columns=Columns.desc_cols.value)
    X_test = svd.transform(X_test)
    test_desc = pd.DataFrame(X_test, columns=Columns.desc_cols.value)
    return train_desc, test_desc

def label_encoder(arr, cols):
    enc = LabelEncoder()
    for col in cols:
        enc_labels = enc.fit_transform(arr[col])

        arr[col] = enc_labels
    return arr


def prepare_data(train, test):

    train["DescScore"].fillna(0, inplace=True)
    train["DescMagnitude"].fillna(0, inplace=True)
    test["DescScore"].fillna(0, inplace=True)
    test["DescMagnitude"].fillna(0, inplace=True)
    train["Description"].fillna("none", inplace=True)
    test["Description"].fillna("none", inplace=True)
    enc = LabelEncoder()
    train["RescuerID"] = enc.fit_transform(train["RescuerID"])
    test["RescuerID"] = enc.fit_transform(test["RescuerID"])
    train["DescLength"] = train["Description"].str.len()
    test["DescLength"] = test["Description"].str.len()
    #train[Columns.ind_num_cat_columns.value] = train[Columns.ind_num_cat_columns.value].astype('category')
    #test[Columns.ind_num_cat_columns.value] = train[Columns.ind_num_cat_columns.value].astype('category')
    # train = conv_cat_variable(train)
    # test = conv_cat_variable(test)
    train_x = train[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value]
    #print(train_x)
    train_y = train[Columns.dep_columns.value]
    test_x = test[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value]
    test_id = test[Columns.iden_columns.value]
    train_desc, test_desc = tfidf(train, test, Columns.n_desc_svdcomp.value)
    train_x = pd.concat([train_x, train_desc], axis=1)
    test_x = pd.concat([test_x, test_desc], axis=1)
    #train_x, test_x = scale_num_var(train_x, test_x)
    return train_x, train_y, test_x, test_id


def create_dict(file, idx, col):
    df = pd.read_csv(file)
    df.set_index(idx, inplace = True)
    dct = df[col].to_dict()
    dct[0] = "Not Set"
    return dct

def scale_num_var(train, test):
    for col in Columns.ind_cont_columns.value:
        train[col] = (train[col]-train[col].mean())/train[col].std()
        test[col] = (test[col]-train[col].mean())/train[col].std()
    return train, test

def conv_cat_variable(df):
    dct1 = create_dict(Paths.base.value+"color_labels.csv", "ColorID", "ColorName")
    df["Color1"] = df["Color1"].map(dct1)
    df["Color2"] = df["Color2"].map(dct1)
    df["Color3"] = df["Color3"].map(dct1)
    dct2 = create_dict(Paths.base.value + "breed_labels.csv", "BreedID", "BreedName")
    df["Breed1"] = df["Breed1"].map(dct2)
    df["Breed2"] = df["Breed2"].map(dct2)
    dct3 = create_dict(Paths.base.value + "state_labels.csv", "StateID", "StateName")
    df["State"] = df["State"].map(dct3)
    # dict for type
    dct4 = {1: "Dog", 2: "Cat"}
    df["Type"] = df["Type"].map(dct4)
    return df

if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    #tfidf(train, test)
    train_x, train_y, test_x, test_id = prepare_data(train,test)

    print(train_x.describe())
