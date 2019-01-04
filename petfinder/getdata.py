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
import os
import json
from scipy.stats import ttest_ind, f_oneway, normaltest, ks_2samp

def read_data():
    train = pd.read_csv("C:/datasets/petfinder.my/train/train.csv")
    test = pd.read_csv("C:/datasets/petfinder.my/test/test.csv")
    train_snt = get_desc_anly("train")
    test_snt = get_desc_anly("test")

    train = train.set_index("PetID").join(train_snt.set_index("PetID")).reset_index()
    test = test.set_index("PetID").join(test_snt.set_index("PetID")).reset_index()
    return train, test


def get_desc_anly(type):
    if type == "train":
        path = "C:/datasets/petfinder.my/train_sentiment/"#../input/train_sentiment/
    else:
        path = "C:/datasets/petfinder.my/test_sentiment/"#../input/test_sentiment/

    files = [f for f in os.listdir(path) if (f.endswith('.json') & os.path.isfile(path+f))]

    df = pd.DataFrame(columns=["PetID", "DescScore", "DescMagnitude"])
    i = 0
    for f in files:
        #print(path + f)
        with open(path+f, encoding="utf8") as json_data:
            data = json.load(json_data)
        #print(data)
        #pf = pd.DataFrame.from_dict(data, orient='index').T.set_index('index')

        #print(data["documentSentiment"]["score"],data["documentSentiment"]["magnitude"])
        df.loc[i]= [f[:-5],data["documentSentiment"]["score"],data["documentSentiment"]["magnitude"]]
        i = i+1

    #print(df)
    if type == "train":
        path = "C:/datasets/petfinder.my/train/sentiment.csv" #../input/train/sentiment.csv
    else:
        path = "C:/datasets/petfinder.my/test/sentiment.csv" #../input/test/sentiment.csv
    return df


if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #train, test = read_data()
    #print(train.corr())
    train = pd.read_csv("C:/datasets/petfinder.my/train/train.csv")
    args = []

    #sys.exit()








