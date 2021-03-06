from petfinder.get_explore import read_data, Columns, Paths
from petfinder.preprocessing import prepare_data
from petfinder.tools import tfidf_2, plog, detect_outliers, auto_features, auto_adp_features, tsne
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import featuretools as ft
import numpy as np
import sys
from random import randint
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from petfinder.train_regressor import OptimizedRounder

def setItemType(val, c):
    if c in ["Color1", "Color2", "Color3"]:
        v1, v2, v3, v4, v5 = 100, 200, 350, 500, 1000
    elif c in ["RescuerID"]:
        v1, v2, v3, v4, v5 = 1, 5, 10, 20, 50
    elif c in ["Breed1", "Breed2"]:
        v1, v2, v3, v4, v5 = 1, 10, 50, 250, 1000
    elif c in ["Pet_Breed"]:
        v1, v2, v3, v4, v5 = 100, 300, 500, 1500, 3500
    else:
        v1, v2, v3, v4, v5 = 1, 5, 10, 20, 50
    if val > v5:
        return 5
    elif val > v4:
        return 4
    elif val > v3:
        return 3
    elif val > v2:
        return 2
    elif val > v1:
        return 1
    else:
        return 0

def itemType(df, col):
    train_r = df.groupby(col)["PetID"].count().reset_index()
    train_r[col+"_Type"] = train_r.apply(lambda x: setItemType(x['PetID'], col), axis=1)
    #train_r[col+"_Type"] = train_r['PetID']
    return train_r[[col, col+"_Type"]]

def setItemAdp(val):
    if val > 3.5:
        return 4
    elif val > 2.5:
        return 3
    elif val > 1.5:
        return 2
    elif val > 0.5:
        return 1
    else:
        return 0

def itemAdp(df, col):
    train_r = df.groupby(col)["AdoptionSpeed"].mean().reset_index()
    train_r[col+"_Adp"] = train_r.apply(lambda x: setItemAdp(x['AdoptionSpeed']), axis=1)
    #train_r[col+"_Adp"] = train_r['AdoptionSpeed']
    return train_r[[col, col+"_Adp"]]

def set_pet_breed(b1, b2):
    #print(b1, b2)
    if (b1 in  (0, 307)) & (b2 in  (0, 307)):
        return 4
    elif (b1 ==  307) & (b2 not in  (0, 307)):
        return 3
    elif (b2 ==  307) & (b1 not in  (0, 307)):
        return 3
    elif (b1 not in  (0, 307)) & (b2 not in  (0, 307)) & (b1 != b2):
        return 2
    elif (b1 == 0) & (b2 not in  (0, 307)):
        return 1
    elif (b2 == 0) & (b1 not in  (0, 307)):
        return 1
    elif (b1 not in  (0, 307)) & (b2 not in  (0, 307)) & (b1 == b2):
        return 0
    else:
        return 3

def set_mean_fee(df, col):
    dfm = df[[col, "Fee"]].groupby(col).mean().reset_index()
    train = df.set_index(col).join(dfm.set_index(col), rsuffix="_"+col).reset_index()
    return train

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
    print(x_0.shape)
    print(x_1.shape)
    print(x_2.shape)
    print(x_3.shape)
    print(x_4.shape)
    df = pd.DataFrame()
    rep_x = ""
    rep_y = ""
    for i in range(0,5):
        plog("autoenc for " + str(i))
        if i == 0:
            x_norm = np.concatenate((x_1, x_2, x_3, x_4), axis=0)
            x_class = x_0
        elif i == 1:
            x_norm = np.concatenate((x_0, x_2, x_3, x_4), axis=0)
            x_class = x_1
        elif i == 2:
            x_norm = np.concatenate((x_0, x_1, x_3, x_4), axis=0)
            x_class = x_2
        elif i == 3:
            x_norm = np.concatenate((x_0, x_1, x_2, x_4), axis=0)
            x_class = x_3
        elif i == 4:
            x_norm = np.concatenate((x_0, x_1, x_2, x_3), axis=0)
            x_class = x_4

        autoencoder.fit(x_norm, x_norm,
                        batch_size=256, epochs=10,
                        shuffle=True, validation_split=0.20);

        hidden_representation = Sequential()
        hidden_representation.add(autoencoder.layers[0])
        hidden_representation.add(autoencoder.layers[1])
        hidden_representation.add(autoencoder.layers[2])

        norm_hid_rep = hidden_representation.predict(x_norm)
        class_hid_rep = hidden_representation.predict(x_class)

        if rep_x == "":
            rep_x = class_hid_rep
        else :
            rep_x = np.append(rep_x, class_hid_rep, axis=0)

        y_c = np.full(class_hid_rep.shape[0], i)
        if rep_y == "":
            rep_y = y_c
        else:
            rep_y = np.append(rep_y, y_c)

    tsne(rep_x, rep_y)
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
    train.drop(['Lbl_Scr_3','Lbl_Img','Lbl_Scr_2','Lbl_Scr_1'], axis=1, inplace=True)
    test.drop(['Lbl_Scr_3','Lbl_Img','Lbl_Scr_2','Lbl_Scr_1'], axis=1, inplace=True)


    plog("setting length of description and name on train and test")
    train["DescLength"] = train["Description"].str.len()
    train["NameLength"] = train["Name"].str.len()
    test["DescLength"] = test["Description"].str.len()
    test["NameLength"] = test["Description"].str.len()
    plog("Setted length of description and name on train and test")

    train.drop(["Description", "Name"], axis=1, inplace=True)
    test.drop(["Description", "Name"], axis=1, inplace=True)

    plog("Creating pet_breed feature on train dataset")
    train["Pet_Breed"] = train.apply(lambda x: set_pet_breed(x['Breed1'], x['Breed2']), axis=1)
    plog("Pet_breed feature on train dataset created")
    plog("Creating pet_breed feature on test dataset")
    test["Pet_Breed"] = test.apply(lambda x: set_pet_breed(x['Breed1'], x['Breed2']), axis=1)
    plog("Pet_breed feature on test dataset created")

    for c in Columns.fee_mean_incols.value:
        if "Fee_" + c in train.columns.values.tolist():
            train.drop("Fee_" + c, axis=1, inplace=0)
        plog("Creating Fee_" + c + " on train dataset")
        train = set_mean_fee(train, c)
        plog("Created_" + c + " on train dataset")
        if "Fee_" + c in test.columns.values.tolist():
            test.drop("Fee_" + c, axis=1, inplace=0)
        plog("Creating Fee_" + c + " on test dataset")
        test = set_mean_fee(test, c)
        plog("Created_" + c + " on test dataset")

    for c in Columns.item_type_incols.value:
        # if c in train_df.columns.values:
        #    plog("Deleting "+c)
        #    train_df.drop(c, axis=1, inplace=True)
        plog("Creating " + c + "_Type for train on " + c)
        df_itr = itemType(train, c)
        train= train.set_index(c).join(df_itr.set_index(c)).reset_index()
        plog("Created " + c + "_Type for train on " + c)
        plog("Creating " + c + "_Type for test on " + c)
        df_its = itemType(test, c)
        test = test.set_index(c).join(df_its.set_index(c)).reset_index()
        plog("Created " + c + "_Type for test on " + c)

    adpf = 0
    if adpf == 1:
        for c in Columns.item_type_cols.value:
            if c + "_Adp" in train.columns.values:
                plog("Deleting train " + c + "_Adp")
                train.drop([c + "_Adp"], axis=1, inplace=True)
            plog("Creating " + c + "_Adp for train on " + c)
            df_itr = itemAdp(train, c)
            train = train.set_index(c).join(df_itr.set_index(c)).reset_index()
            optR = OptimizedRounder()
            optR.fit(train[c + "_Adp"], train["AdoptionSpeed"])
            coefficients = optR.coefficients()
            pred_test_y_k = optR.predict(train[c + "_Adp"], coefficients)
            train[c+"_Adp"]= pred_test_y_k
            plog("Created " + c + "_Adp for train on " + c)
            if c + "_Adp" in test.columns.values:
                plog("Deleting test " + c + "_Adp")
                test.drop([c + "_Adp"], axis=1, inplace=True)
            plog("Creating " + c + "_Adp for test on " + c)
            # df_its = itemAdp(test_df, c)
            #print(df_itr)
            test = test.set_index(c).join(df_itr.set_index(c)).reset_index()
            test[c + "_Adp"].fillna(3, inplace=True)
            plog("Created " + c + "_Adp for test on " + c)

    autof = 0
    if autof == 1:
        plog("creating new features on train using featuretools")
        train = auto_features(train,
                              Columns.iden_columns.value + Columns.ft_cat_cols.value +
                              Columns.item_type_cols.value + Columns.ft_new_cols.value,
                              Columns.item_type_cols.value + Columns.ft_cat_cols.value)
        plog("created new features on train using featuretools")

        plog("creating new features on test using featuretools")
        test = auto_features(test,
                             Columns.iden_columns.value + Columns.ft_cat_cols.value +
                             Columns.item_type_cols.value + Columns.ft_new_cols.value,
                             Columns.item_type_cols.value + Columns.ft_cat_cols.value)
        plog("created new features on test using featuretools")

        plog("creating adoptionspeed based new features on train and test using featuretools")
        train, test = auto_adp_features(train, test,
                                        Columns.iden_columns.value + Columns.ft_cat_cols.value +
                                        Columns.item_type_cols.value + ["AdoptionSpeed"],
                                        Columns.item_type_cols.value + Columns.ft_cat_cols.value)
        plog("Created adoptionspeed based new features on train and test using featuretools")

    train.drop(["RescuerID"], axis=1, inplace=True)
    test.drop(["RescuerID"], axis=1, inplace=True)

    print("-------train dataset", train.shape, "columns-------")
    #print(train.columns.values)
    print("-------test dataset", test.shape, "columns-------")
    #print(test.columns.values)
    return train, test

def filter_by_varth(train, test, threshold):
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(train)
    indices = vt.get_support(indices=True)
    return(train.iloc[:,indices], test.iloc[:,indices])

def finalize_data(train, test):

    train_x = train[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
                    + Columns.desc_cols.value + Columns.img_num_cols_1.value
                    + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_cols.value
                    + Columns.ft_cols.value + Columns.item_type_cols.value + Columns.fee_mean_cols.value]
    train_y = train[Columns.dep_columns.value]
    test_x = test[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
                  + Columns.desc_cols.value + Columns.img_num_cols_1.value
                  + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_cols.value
                  + Columns.ft_cols.value + Columns.item_type_cols.value + Columns.fee_mean_cols.value]
    test_id = test[Columns.iden_columns.value]

    return train_x, train_y, test_x, test_id

def recursive_feature_removal():
    pass

def group_x_by_y(df, x, y, type):
    df_g = df.groupby(y).agg({x:type})
    df_g.rename({x: x+"_"+type+"_by_"+y}, axis='columns', inplace=True)
    df = df.set_index(y).join(df_g).reset_index()
    return df

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    train = pd.read_csv(Paths.base.value + "train/train.csv")

    print(group_x_by_y(train, "Age", "Breed1", "std"))

    sys.exit()


