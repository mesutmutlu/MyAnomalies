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
from petfinder.tools import  label_encoder, scale_num_var, tfidf_2, detect_outliers
import os

def setRescuerType(val):
    if val >= 10:
        return 3
    elif val > 5:
        return 2
    elif val > 1:
        return 1
    else:
        return 0

def rescuerType(df):
    train_r = df.groupby(['RescuerID'])["PetID"].count().reset_index()
    #train_r.columns = train_r.columns.droplevel()
    # print(h.sort_values(by=['count'],ascending=False))
    train_r["RescuerType"] = train_r.apply(lambda x: setRescuerType(x['PetID']), axis=1)
    #train_r.columns = ["RescuerID", "Count"]
    return train_r[["RescuerID", "RescuerType"]]

def setStateAdoptionSpeed(val):
    if val >= 3:
        return 3
    elif val > 2.5:
        return 2
    elif val > 2:
        return 1
    else:
        return 0

def stateAdoptionSpeed(df):
    train_r = df.groupby(['State'])["AdoptionSpeed"].mean().reset_index()
    train_r["StateAdoptionSpeed"] = train_r.apply(lambda x: setStateAdoptionSpeed(x['AdoptionSpeed']), axis=1)
    return train_r[["State", "StateAdoptionSpeed"]]

def quantile_bin(df, col):
    quantile_list = [0, .25, .5, .75, 1.]
    quantiles = df[col].quantile(quantile_list)
    return quantiles


def prepare_data(train, test):

    train["Description"].fillna("", inplace=True)
    svd_train_desc = tfidf_2(train["Description"], Columns.n_desc_svdcomp.value, Columns.desc_cols.value)
    train["Lbl_Dsc"].fillna("", inplace=True)
    svd_train_lbldsc = tfidf_2(train["Lbl_Dsc"], Columns.n_iann_svdcomp.value, Columns.iann_cols.value)
    train = pd.concat([train, svd_train_desc, svd_train_lbldsc], axis=1)

    test["Description"].fillna("", inplace=True)
    svd_test_desc = tfidf_2(test["Description"], Columns.n_desc_svdcomp.value, Columns.desc_cols.value)
    test["Lbl_Dsc"].fillna("", inplace=True)
    svd_test_lbldsc = tfidf_2(test["Lbl_Dsc"], Columns.n_iann_svdcomp.value, Columns.iann_cols.value)
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
    print("quantile binning fee")

    print("setting state adoption mean")
    df_ast = stateAdoptionSpeed(train)
    train = train.set_index("State").join(df_ast.set_index("State")).reset_index()
    #f_rts = stateAdoptionSpeed(test)
    test = test.set_index("State").join(df_ast.set_index("State")).reset_index()
    print("defining rescuer type")
    df_rtt = rescuerType(train)
    train = train.set_index("RescuerID").join(df_rtt.set_index("RescuerID")).reset_index()
    df_rts = rescuerType(test)
    test = test.set_index("RescuerID").join(df_rts.set_index("RescuerID")).reset_index()
    print("rescuerid encoding")
    enc = LabelEncoder()
    train["RescuerID"] = enc.fit_transform(train["RescuerID"])
    test["RescuerID"] = enc.fit_transform(test["RescuerID"])
    drop_cols = Columns.ind_text_columns.value + Columns.img_lbl_cols_1.value + Columns.img_lbl_cols_2.value + \
           Columns.img_lbl_cols_3.value + Columns.img_lbl_col.value
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    df_o = detect_outliers(train, 0)
    train = pd.concat([train, df_o], axis=1)
    train = train[train["outlier"] == 1]
    train_x = train[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
                       + Columns.desc_cols.value + Columns.img_num_cols_1.value
                       + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_cols.value]
    train_y = train[Columns.dep_columns.value]
    test_x = test[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
                     + Columns.desc_cols.value + Columns.img_num_cols_1.value
                     + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_cols.value]
    test_id = test[Columns.iden_columns.value]
    #train_x, test_x = scale_num_var(train_x, test_x)
    return train_x, train_y, test_x, test_id


if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    #pdf = stateAdoptionSpeed(train)
    #print(pdf)
    #sys.exit()
    #pdf.hist()
    #plt.show()
    #tfidf(train, test)
    #sys.exit()
    train_x, train_y, test_x, test_id = prepare_data(train,test)
    print(train_x.head())
    print(test_x.head())
    sys.exit()
    #print(train_x.columns.values)
    #print(test_x.columns.values)
    data = pd.concat([train_x, train_y], axis=1)
    print(data[data["AdoptionSpeed"] == 2])
    ax = sns.scatterplot(x="DescLength", y="DescScore", hue="AdoptionSpeed", data=data[data["AdoptionSpeed"] == 2])
    plt.show()
    print(train_x.shape)
    print(train_x.columns.values)
    print(train_y.shape)
    print(train_y.columns.values)
    print(test_x.shape)
    print(test_x.columns.values)
    print(test_id.shape)
    print(test_id.columns.values)

    #print(train_x.head())
