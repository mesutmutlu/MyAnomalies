from petfinder.get_explore import read_data
from sklearn.preprocessing import LabelEncoder
from enum import Enum
from petfinder.get_explore import Columns, Paths
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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