from petfinder.get_explore import read_data
from sklearn.preprocessing import LabelEncoder
from enum import Enum
from petfinder.get_explore import Columns, Paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

def tfidf(train , test, col, n_comp):
    train_desc = train[col]
    test_desc = test[col]

    tfv = TfidfVectorizer(min_df=2, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          )

    # Fit TFIDF
    tfv.fit(list(train_desc))
    X = tfv.transform(train_desc)
    X_test = tfv.transform(test_desc)

    svd = TruncatedSVD(n_components=n_comp)
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

    svd = TruncatedSVD(n_components=n_comp)
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


def scale_num_var(train, test):
    for col in Columns.ind_cont_columns.value:
        train[col] = (train[col]-train[col].mean())/train[col].std()
        test[col] = (test[col]-train[col].mean())/train[col].std()
    return train, test


def fill_na(arr, cols, val):
    for col in cols:
        arr[col].fillna(val, inplace=True)
    return arr


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