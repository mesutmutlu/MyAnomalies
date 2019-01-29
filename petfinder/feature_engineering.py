from petfinder.get_explore import read_data, Columns, Paths
from petfinder.preprocessing import prepare_data
from petfinder.tools import tfidf_2, plog, detect_outliers, auto_features
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import featuretools as ft
import numpy as np
import sys
from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from numpy import argmax, array_equal
from keras.models import Model, Sequential
from random import randint
import pandas as pd
from keras import regularizers
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

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
    train_r["RescuerType"] = train_r.apply(lambda x: setRescuerType(x['PetID']), axis=1)
    return train_r[["RescuerID", "RescuerType"]]


def quantile_bin(df, col):
    quantile_list = [0, .25, .5, .75, 1.]
    quantiles = df[col].quantile(quantile_list)
    return quantiles

def autoenc(data):
    ## input layer
    df = data
    X = df.drop(['AdoptionSpeed'], axis=1).values
    Y = df["AdoptionSpeed"].values
    input_layer = Input(shape=(X.shape[1],))

    ## encoding part
    encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoded = Dense(50, activation='relu')(encoded)

    ## decoding part
    decoded = Dense(50, activation='tanh')(encoded)
    decoded = Dense(100, activation='tanh')(decoded)

    ## output layer
    output_layer = Dense(X.shape[1], activation='relu')(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adadelta", loss="mse")

    x = data.drop(["AdoptionSpeed"], axis=1)
    y = data["AdoptionSpeed"].values

    x_scale = preprocessing.MinMaxScaler().fit_transform(x.values)
    x_0, x_1, x_2, x_3, x_4 = x_scale[y == 0], x_scale[y == 1], x_scale[y == 2], x_scale[y == 3], x_scale[y == 4]
    df = pd.DataFrame()
    for x in [x_0, x_1, x_2, x_3, x_4]:
        x_norm = [x_0, x_1, x_2, x_3, x_4].remove(x)
        x_fraud = x
        autoencoder.fit(x_norm, x_norm,
                        batch_size=256, epochs=10,
                        shuffle=True, validation_split=0.20);

        hidden_representation = Sequential()
        hidden_representation.add(autoencoder.layers[0])
        hidden_representation.add(autoencoder.layers[1])
        hidden_representation.add(autoencoder.layers[2])

        norm_hid_rep = hidden_representation.predict(x_norm[:3000])
        fraud_hid_rep = hidden_representation.predict(x_fraud)

        y_n = np.zeros(norm_hid_rep.shape[0])
        y_f = np.ones(fraud_hid_rep.shape[0])
        rep_y = np.append(y_n, y_f)
        if len(df) == 0:
            df = rep_y
        else:
            df = pd.concat([df, rep_y])
    print(df)
    #sns.scatterplot(x="tsne1", y="tsne2", hue="AdoptionSpeed", data=data, legend="full")

def add_features(train, test):
    plog("tfidf for train description started")
    svd_train_desc = tfidf_2(train["Description"], Columns.n_desc_svdcomp.value, Columns.desc_cols.value)
    plog("tfidf for train description ended")
    plog("tfidf for train image label started")
    svd_train_lbldsc = tfidf_2(train["Lbl_Img"], Columns.n_iann_svdcomp.value, Columns.iann_cols.value)
    plog("tfidf for train image label ended")
    train = pd.concat([train, svd_train_desc, svd_train_lbldsc], axis=1)
    plog("train tfidf concatenation ended")

    plog("tfidf for test description started")
    svd_test_desc = tfidf_2(test["Description"], Columns.n_desc_svdcomp.value, Columns.desc_cols.value)
    plog("tfidf for test description ended")
    plog("tfidf for test image label started")
    svd_test_lbldsc = tfidf_2(test["Lbl_Img"], Columns.n_iann_svdcomp.value, Columns.iann_cols.value)
    plog("tfidf for test image label ended")
    test = pd.concat([test, svd_test_desc, svd_test_lbldsc], axis=1)
    plog("test tfidf concatenation ended")

    plog("setting length of description and name on train and test")
    train["DescLength"] = train["Description"].str.len()
    train["NameLength"] = train["Name"].str.len()
    test["DescLength"] = test["Description"].str.len()
    test["NameLength"] = test["Description"].str.len()
    plog("Setted length of description and name on train and test")

    plog("creating rescuerType for train")
    df_rtt = rescuerType(train)
    train = train.set_index("RescuerID").join(df_rtt.set_index("RescuerID")).reset_index()
    plog("rescuerType for train created")

    plog("creating rescuerType for test")
    df_rts = rescuerType(test)
    test = test.set_index("RescuerID").join(df_rts.set_index("RescuerID")).reset_index()
    plog("rescuerType for test created")

    plog("creating new features on train using featuretools")
    train = auto_features(train, Columns.ft_cat_cols.value + Columns.ft_new_cols.value, Columns.ft_cat_cols.value)
    plog("created new features on train using featuretools")

    plog("creating new features on test using featuretools")
    test = auto_features(test, Columns.ft_cat_cols.value + Columns.ft_new_cols.value, Columns.ft_cat_cols.value)
    plog("created new features on test using featuretools")

    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    plog("outlier detection started on train")
    o_cols = []
    for oc in Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value + Columns.dep_columns.value +\
    Columns.desc_cols.value + Columns.img_num_cols_1.value + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value +\
    Columns.iann_cols.value + Columns.ft_cols.value:
        if oc not in ["RescuerID"]:
            o_cols.append(oc)
    df_o = detect_outliers(train[o_cols], 1)
    train = pd.concat([train, df_o], axis=1)
    plog("outlier detection ended by removing outliers on train")
    return train, test

def finalize_data(train, test):

    train_x = train[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
                    + Columns.desc_cols.value + Columns.img_num_cols_1.value
                    + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_cols.value
                    + Columns.ft_cols.value + ["outlier"]]
    train_y = train[Columns.dep_columns.value + ["outlier"]]
    test_x = test[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
                  + Columns.desc_cols.value + Columns.img_num_cols_1.value
                  + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_cols.value
                  + Columns.ft_cols.value]
    test_id = test[Columns.iden_columns.value]

    return train_x, train_y, test_x, test_id

def recursive_feature_removal():
    pass


if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    train, test = read_data()
    train, test = prepare_data(train, test)

    train, test = add_features(train, test)
    x_train, y_train, x_test, id_test = finalize_data(train, test)


    x_train_wo = x_train.drop(["outlier"], axis=1)
    print(x_train_wo.shape)
    y_train_wo = y_train.drop(["outlier"], axis=1)
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    x_train_wo2 = sel.fit_transform(x_train_wo)
    print(x_train_wo2.shape)
    sys.exit()

    autoenc(pd.concat([x_train_wo, y_train_wo], axis=1))
    print(x_train.shape)
    print(x_train.columns.values)
    print(y_train.shape)
    print(y_train.columns.values)
    print(x_test.shape)
    print(x_test.columns.values)
    print(id_test.shape)
    print(id_test.columns.values)

    sys.exit()


