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
from petfinder.getdata import read_data
from sklearn.preprocessing import LabelEncoder
from enum import Enum


class Columns(Enum):
    ind_cont_columns = ["Age", "Fee", "VideoAmt", "PhotoAmt", "DescScore", "DescMagnitude"]
    ind_num_cat_columns = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize",
                           "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "State",
                           "RescuerID"]
    ind_text_columns = ["Name", "Description"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]


def fill_na(arr, cols, val):
    for col in cols:
        arr[col].fillna(val, inplace=True)
    return arr


def label_encoder(arr, cols):
    enc = LabelEncoder()
    for col in cols:
        enc_labels = enc.fit_transform(arr[col])

        arr[col] = enc_labels
    return arr


def prepare_data():

    tr1, te1  = read_data()
    tr2 = fill_na(tr1, ["DescScore", "DescMagnitude"], 0)
    train = label_encoder(tr2, ["RescuerID"])
    te2 = fill_na(te1, ["DescScore", "DescMagnitude"], 0)
    test = label_encoder(te2, ["RescuerID"])
    ind_cont_columns = ["Age", "Fee","VideoAmt", "PhotoAmt", "DescScore", "DescMagnitude"]
    ind_num_cat_columns = ["Type","Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3","MaturitySize","FurLength",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "State", "RescuerID"]
    ind_text_columns = ["Name", "Description"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]
    train_x = train[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value]
    train_y = train[Columns.dep_columns.value]
    test_x = test[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value]
    test_id = test[Columns.iden_columns.value]
    return train_x, train_y, test_x, test_id


if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #train, test = read_data()
    train_x, train_y, test_x, test_id = prepare_data()
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_id)