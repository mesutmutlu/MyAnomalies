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

def read_data():
    train = pd.read_csv("C:/datasets/petfinder.my/train/train.csv")
    test = pd.read_csv("C:/datasets/petfinder.my/test/test.csv")
    return train, test

def prepare_data():

    train, test  = read_data()
    ind_cont_columns = ["Age", "Fee","VideoAmt", "PhotoAmt"]
    ind_num_cat_columns = ["Type","Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3","MaturitySize","FurLength",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "State"]
    ind_cat_conv_columns = ["RescuerID"]
    ind_text_columns = ["Name", "Description"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]
    train_x = train[ind_cont_columns + ind_num_cat_columns]
    train_y = train[dep_columns]
    test_x = test[ind_cont_columns + ind_num_cat_columns]
    test_id = test[iden_columns]
    return train_x, train_y, test_x, test_id

def get_desc_anal(df, iden_columns):
    path = "C:/datasets/petfinder.my/train_sentiment/"

    files = [f for f in os.listdir(path) if (f.endswith('.json') & os.path.isfile(path+f))]

    df = pd.DataFrame(columns=["pet_id", "score", "magnitude"])
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

    print(df)

def get_img_anal(df, iden_columns):
    path = "C:/datasets/petfinder.my/train_metadata/"

    files = [f for f in os.listdir(path) if (f.endswith('.json') & os.path.isfile(path+f))]

    df = pd.DataFrame(columns=["pet_id", "score", "magnitude"])
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

    print(df)



if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()

    #print(train.head())
    #print(train.describe(include="all"))
    train_x, train_y, test_x, test_id = prepare_data()
    ind_cont_columns = ["Age", "Fee","VideoAmt", "PhotoAmt"]
    ind_num_cat_columns = ["Type","Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3","MaturitySize","FurLength",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "State"]
    ind_cat_conv_columns = ["RescuerID"]
    ind_text_columns = ["Name", "Description"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]

    get_desc_anal(train, iden_columns)

