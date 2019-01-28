from sklearn import manifold
from petfinder.get_explore import read_data, Columns, Paths
from petfinder.preprocessing import prepare_data
from petfinder.tools import tfidf_2, plog, detect_outliers
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime

def tsne():
    pass

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
    df = df.set_index("RescuerID").join(train_r.set_index("RescuerID")).reset_index()
    return df


def quantile_bin(df, col):
    quantile_list = [0, .25, .5, .75, 1.]
    quantiles = df[col].quantile(quantile_list)
    return quantiles

def add_features(train, test):
    plog("tfidf for train set started")
    svd_train_desc = tfidf_2(train["Description"], Columns.n_desc_svdcomp.value, Columns.desc_cols.value)
    svd_train_lbldsc = tfidf_2(train["Lbl_Img"], Columns.n_iann_svdcomp.value, Columns.iann_cols.value)
    train = pd.concat([train, svd_train_desc, svd_train_lbldsc], axis=1)
    plog("tfidf for train set ended")

    plog("tfidf for test set started")
    svd_test_desc = tfidf_2(test["Description"], Columns.n_desc_svdcomp.value, Columns.desc_cols.value)
    svd_test_lbldsc = tfidf_2(test["Lbl_Img"], Columns.n_iann_svdcomp.value, Columns.iann_cols.value)
    test = pd.concat([test, svd_test_desc, svd_test_lbldsc], axis=1)
    plog("tfidf for test set ended")

    plog("setting length of description and name")
    train["DescLength"] = train["Description"].str.len()
    train["NameLength"] = train["Name"].str.len()
    test["DescLength"] = test["Description"].str.len()
    test["NameLength"] = test["Description"].str.len()
    plog("length of description and name setted")

    plog("creating rescuerType for train")
    df_rtt = rescuerType(train)
    train = train.set_index("RescuerID").join(df_rtt.set_index("RescuerID")).reset_index()
    plog("rescuerType for train created")
    plog("creating rescuerType for test")
    df_rts = rescuerType(test)
    test = test.set_index("RescuerID").join(df_rts.set_index("RescuerID")).reset_index()
    plog("rescuerType for test created")

    plog("outlier detection started")
    df_o = detect_outliers(train, 0)
    train = pd.concat([train, df_o], axis=1)
    train = train[train["outlier"] == 1]
    plog("outlier detection ended by removing outliers")

    drop_cols = Columns.ind_text_columns.value + Columns.img_lbl_cols_1.value + Columns.img_lbl_cols_2.value + \
                Columns.img_lbl_cols_3.value + Columns.img_lbl_col.value
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    train_x = train[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
                    + Columns.desc_cols.value + Columns.img_num_cols_1.value
                    + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_cols.value]
    train_y = train[Columns.dep_columns.value]
    test_x = test[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value
                  + Columns.desc_cols.value + Columns.img_num_cols_1.value
                  + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_cols.value]
    test_id = test[Columns.iden_columns.value]
    # train_x, test_x = scale_num_var(train_x, test_x)
    return train_x, train_y, test_x, test_id


if __name__ == "__main__":

    train, test = read_data()
    print(len(train))
    x_train, y_train, x_test, id_test = prepare_data(train, test)
    perplexities = [5, 30, 50, 100]
    #(fig, subplots) = plt.subplots(1, 5, figsize=(15, 8))
    print(len(x_train),len(y_train))
#    ax = subplots[0][0]
    print("tsne started")

    for i, perplexity in enumerate(perplexities):
        #ax = subplots[0][i + 1]

        t0 = time()
        print(datetime.datetime.now())
        tsne = manifold.TSNE(n_components=2, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(x_train)
        print(datetime.datetime.now())
        t1 = time()
        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        print(len(x_train), len(Y))
        print(Y)
        data = pd.concat([pd.DataFrame(data=Y, columns=["tsne1", "tsne2"]), y_train], axis=1)
        print(data)
        #ax.set_title("Perplexity=%d" % perplexity)
        sns.scatterplot(x="tsne1", y="tsne2", hue="AdoptionSpeed",  data=data, legend="full")
        plt.show()

        #ax.axis('tight')