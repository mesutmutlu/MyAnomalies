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
from petfinder.tools import tfidf, label_encoder, scale_num_var, tfidf_2


def prepare_data(train, test):
    train["Description"].fillna("", inplace=True)
    svd_train_desc = tfidf_2(train["Description"], 70, Columns.desc_cols.value)
    train["Lbl_Dsc"].fillna("", inplace=True)
    svd_train_lbldsc = tfidf_2(train["Lbl_Dsc"], 5, Columns.iann_cols.value)
    train = pd.concat([train, svd_train_desc, svd_train_lbldsc], axis=1)

    test["Description"].fillna("", inplace=True)
    svd_test_desc = tfidf_2(test["Description"], 70, Columns.desc_cols.value)
    test["Lbl_Dsc"].fillna("", inplace=True)
    svd_test_lbldsc = tfidf_2(test["Lbl_Dsc"], 5, Columns.iann_cols.value)
    test = pd.concat([test, svd_test_desc, svd_test_lbldsc], axis=1)


    train["Name"].fillna("", inplace=True)
    test["Name"].fillna("", inplace=True)
    train["DescLength"] = train["Description"].str.len()
    train["NameLength"] = train["Name"].str.len()
    test["DescLength"] = test["Description"].str.len()
    test["NameLength"] = test["Description"].str.len()
    train["DescScore"].fillna(0, inplace=True)
    train["DescMagnitude"].fillna(0, inplace=True)
    test["DescScore"].fillna(0, inplace=True)
    test["DescMagnitude"].fillna(0, inplace=True)

    print("rescuerid encoding")
    enc = LabelEncoder()
    train["RescuerID"] = enc.fit_transform(train["RescuerID"])
    test["RescuerID"] = enc.fit_transform(test["RescuerID"])
    drop_cols = Columns.ind_text_columns.value + Columns.img_lbl_cols_1.value + Columns.img_lbl_cols_2.value + \
           Columns.img_lbl_cols_3.value + Columns.img_lbl_col.value
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)

    train_x = train.drop(Columns.dep_columns.value, axis=1)
    train_y = train[Columns.dep_columns.value]
    test_x = test.drop(Columns.iden_columns.value, axis=1)
    test_id = test[Columns.iden_columns.value]
    #train_x, test_x = scale_num_var(train_x, test_x)
    train_x.fillna(-1, inplace=True)
    test_x.fillna(-1, inplace=True)
    return train_x, train_y, test_x, test_id


if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    #tfidf(train, test)
    train_x, train_y, test_x, test_id = prepare_data(train,test)
    print(train_x.describe())
    print(test_x.describe())

    #print(train_x.head())
