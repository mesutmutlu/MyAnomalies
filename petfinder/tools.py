from petfinder.get_explore import read_data
from sklearn.preprocessing import LabelEncoder
from enum import Enum
from petfinder.get_explore import Columns, Paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from nltk.corpus import stopwords
import nltk

def create_dict(file, idx, col):
    df = pd.read_csv(file)
    df.set_index(idx, inplace = True)
    dct = df[col].to_dict()
    dct[0] = "Not Set"
    return dct

def tfidf(train , test, col, n_comp, cols):
    stop_words = set(stopwords.words('english'))
    print(type(stop_words))
    print("starting tfidf for train and test set")
    train_desc = train[col].fillna("none")
    test_desc = test[col].fillna("none")

    tfv = TfidfVectorizer(min_df=2, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words=stop_words
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
    train_desc = pd.DataFrame(X, columns=cols)
    X_test = svd.transform(X_test)
    test_desc = pd.DataFrame(X_test, columns=cols)

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
    train_desc = pd.DataFrame(X, columns=cols)
    X_test = svd.transform(X_test)
    test_desc = pd.DataFrame(X_test, columns=cols)
    return train_desc, test_desc


def tfidf_2(train, n_comp, out_cols):
    print("starting tfidf")
    stop_words = set(stopwords.words('english'))
    tfv = TfidfVectorizer(min_df=2, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words=stop_words)

    # Fit TFIDF
    tfv.fit(list(train))
    X = tfv.transform(train)

    svd = TruncatedSVD(n_components=n_comp)
    svd.fit(X)
    # print(svd.explained_variance_ratio_.sum())
    # print(svd.explained_variance_ratio_)
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

if __name__ == "__main__":

    nltk.download()