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
from petfinder.tools import detect_outliers, plog

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    train, test = prepare_data(train, test)
    train, test = add_features(train, test)

    ccols = train.columns.values.tolist()
    for c in ccols:
        if (train[c].isna().any()) & \
                (c in Columns.img_num_cols_1.value + Columns.img_num_cols_2.value  + Columns.img_num_cols_3.value ):
            plog("filling "+c+" with 0 on train dataset")
            train[c].fillna(0, inplace=True)

    ccols = test.columns.values.tolist()
    for c in ccols:
        if (test[c].isna().any()) & \
                (c in Columns.img_num_cols_1.value + Columns.img_num_cols_2.value  + Columns.img_num_cols_3.value ):
            plog("filling " + c + " with 0 on test dataset")
            test[c].fillna(0, inplace=True)

    o_cols = train.columns.values.tolist()
    for c in ["PetID", "Breed1", "Breed2", "RescuerID", "AdoptionSpeed","Color1", "Color2", "Color3", "State"]:
        if c in o_cols:
            o_cols.remove(c)
    #train[o_cols] = train[o_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
    #train[o_cols] = train[o_cols].apply(lambda x: x.fillna(x.mean()), axis=0)



    plog("Outlier detection started")

    df_o = detect_outliers(train[o_cols + ["AdoptionSpeed"]],1)
    print("df_o", df_o.shape, "train_df", train.shape)
    train = pd.concat([train, df_o], axis=1)
    train.to_csv("train_f.csv")
    test.to_csv("test_f.csv")
    plog("Outlier detection ended")
    if "PetID" in train.columns.values:
        train_pet_id = train["PetID"]
        train_adoption_speed = train["AdoptionSpeed"]
        train.drop(["PetID", "AdoptionSpeed"], axis=1, inplace=True)

    if "PetID" in test.columns.values:
        test_pet_id = test["PetID"]
        test.drop(["PetID"], axis=1, inplace=True)

    #train_t, test_t = filter_by_varth(train, test, (.8 * (1 - .8)))

    #print(train_t.columns.values)
    #print(test_t.columns.values)
    # train_df = pd.concat([train_t, train_pet_id, train_adoption_speed], axis=1)
    # test_df = pd.concat([test_t, test_pet_id], axis=1)


    #train_x, test_x = filter_by_varth(train, test, (.8 * (1 - .8)))
    train["outlier"].fillna(1, inplace=True)
    outlier = 1
    print("uniques")
    print(train["outlier"].unique())
    print(train[train["outlier"]<=outlier].shape)
    x_train, y_train, x_test, id_test = finalize_data(train[train["outlier"]<=outlier], test)

    x_train.to_csv("x_train.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    x_test.to_csv("x_test.csv", index=False)
    id_test.to_csv("id_test.csv", index=False)

    print(x_train.shape)
    print(x_train.columns.values)
    print(y_train.shape)
    print(y_train.columns.values)
    print(x_test.shape)
    print(x_test.columns.values)
    print(id_test.shape)
    print(id_test.columns.values)