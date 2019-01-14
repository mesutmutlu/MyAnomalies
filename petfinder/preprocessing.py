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
from petfinder.tools import tfidf, label_encoder, scale_num_var


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
    train_desc, test_desc = tfidf(train, test, "Description", 120)
    train_x = pd.concat([train_x, train_desc], axis=1)
    test_x = pd.concat([test_x, test_desc], axis=1)
    train_lbl, test_lbl = tfidf(train, test, "Lbl_Dsc", 10)
    train_x2 = train_x.set_index("PetID").join(train_lbl.set_index("PetID"), how="left").reset_index()
    test_x2 = test_x.set_index("PetID").join(test_lbl.set_index("PetID"), how="left").reset_index()
    train_x2.drop(["Lbl_Dsc"], axis=1, inplace=True)
    test_x2.drop(["Lbl_Dsc"], axis=1, inplace=True)
    #train_x, test_x = scale_num_var(train_x, test_x)
    return train_x2, train_y, test_x2, test_id


def create_dict(file, idx, col):
    df = pd.read_csv(file)
    df.set_index(idx, inplace = True)
    dct = df[col].to_dict()
    dct[0] = "Not Set"
    return dct


if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    #tfidf(train, test)
    train_x, train_y, test_x, test_id = prepare_data(train,test)

    print(train_x.head())
