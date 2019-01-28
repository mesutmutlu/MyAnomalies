import pandas as pd
import sys
from petfinder.get_explore import read_data
from sklearn.preprocessing import LabelEncoder


def prepare_data(train, test):

    train["Description"].fillna("", inplace=True)
    test["Description"].fillna("", inplace=True)
    train["Lbl_Img"].fillna("", inplace=True)
    test["Lbl_Img"].fillna("", inplace=True)
    train["Name"].fillna("", inplace=True)
    test["Name"].fillna("", inplace=True)
    train["DescScore"].fillna(0, inplace=True)
    train["DescMagnitude"].fillna(0, inplace=True)
    test["DescScore"].fillna(0, inplace=True)
    test["DescMagnitude"].fillna(0, inplace=True)


    enc = LabelEncoder()
    train["RescuerID"] = enc.fit_transform(train["RescuerID"])
    test["RescuerID"] = enc.fit_transform(test["RescuerID"])

    return train, test


if __name__ == "__main__":
    pass
