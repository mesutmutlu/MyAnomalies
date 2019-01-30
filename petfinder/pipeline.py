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
from petfinder.get_explore import read_data, Columns
from petfinder.preprocessing import prepare_data
from petfinder.feature_engineering import finalize_data, add_features, filter_by_varth
from petfinder.tools import detect_outliers

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    train, test = prepare_data(train, test)
    train, test = add_features(train, test)
    if "PetID" in train.columns.values:
        train_pet_id = train["PetID"]
        train.drop(["PetID"], axis=1, inplace=True)
        train_adoption_speed = train["AdoptionSpeed"]
    if "PetID" in test.columns.values:
        test_pet_id = test["PetID"]
        test.drop(["PetID"], axis=1, inplace=True)

    train_t, test_t = filter_by_varth(train, test, (.8 * (1 - .8)))

    print(train_t.columns.values)
    print(test_t.columns.values)
    # train_df = pd.concat([train_t, train_pet_id, train_adoption_speed], axis=1)
    # test_df = pd.concat([test_t, test_pet_id], axis=1)
    sys.exit()

    # print(train_df.head())
    o_numccatols = Columns.ind_num_cat_columns.value.copy()
    o_numccatols.remove("State")
    o_numccatols.remove("Breed1")
    o_numccatols.remove("Breed2")

    v_cols = Columns.ind_cont_columns.value + o_numccatols + Columns.desc_cols.value + \
             Columns.img_num_cols_1.value + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + \
             Columns.iann_cols.value + Columns.ft_cols.value + Columns.item_type_cols.value

    for c in v_cols:
        if c not in train.columns.values:
            print("na", c)
        if train[c].isna().any():
            print("null", c)
            train[c].fillna(0, inplace=True)
    train_x, test_x = filter_by_varth(train, test, (.8 * (1 - .8)))
    train["outlier"].fillna(0, inplace=True)
    outlier = 0
    print("uniques")
    print(train["outlier"].unique())
    print(train[train["outlier"]<=outlier].shape)
    x_train, y_train, x_test, id_test = finalize_data(train[train["outlier"]<=outlier], test)


    print(x_train.shape)
    print(x_train.columns.values)
    print(y_train.shape)
    print(y_train.columns.values)
    print(x_test.shape)
    print(x_test.columns.values)
    print(id_test.shape)
    print(id_test.columns.values)