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
from enum import Enum
from sklearn.base import BaseEstimator
#from petfinder.preprocessing import prepare_data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import scipy as sp


class Paths(Enum):
    if sys.platform == "linux":
        base = "/home/mesut/kaggle/petfinder.my/"
    else:
        base = "C:/datasets/petfinder.my/"

class Columns(Enum):
    ind_cont_columns = ["Age", "Fee", "VideoAmt", "PhotoAmt", "Quantity",
                        "DescScore", "DescMagnitude", "DescLength"]
    ind_num_cat_columns = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "State", "RescuerID",
                           "FurLength", "MaturitySize"]
    ind_text_columns = ["Name", "Description"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]
    desc_cols = ["svd_" + str(i) for i in range(120)]



def read_data():
    train = pd.read_csv(Paths.base.value+"train/train.csv")
    test = pd.read_csv(Paths.base.value+"test/test.csv")
    calc_anly = 0
    if os.path.exists(Paths.base.value+"train_sentiment.csv"):
        train_snt = pd.read_csv(Paths.base.value + "train_sentiment.csv")
    else:
        train_snt = get_desc_anly("train", calc_anly)
    if os.path.exists(Paths.base.value+"test_sentiment.csv"):
        train_snt = pd.read_csv(Paths.base.value + "test_sentiment.csv")
    else:
        test_snt = get_desc_anly("test", calc_anly)
    train = train.set_index("PetID").join(train_snt.set_index("PetID")).reset_index()
    test = test.set_index("PetID").join(test_snt.set_index("PetID")).reset_index()
    return train, test



def get_desc_anly(type, rc):

    if type == "train":
        path = Paths.base.value+"train_sentiment/"#../input/train_sentiment/
        fpath = Paths.base.value+"train_sentiment/train_sentiment.csv"
    else:
        path = Paths.base.value+"test_sentiment/"#../input/test_sentiment/
        fpath = Paths.base.value + "test_sentiment/test_sentiment.csv"

    if rc == 1 or not(os.path.exists(fpath)):
        if os.path.exists(fpath):
            os.remove(fpath)

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

        df.to_csv(fpath,index=False)
    return pd.read_csv(fpath)


if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #train, test = read_data()
    #print(train.corr())
    #print(sys.platform)
    train, test = read_data()

    print(type(train["Description"].str.len().values))
    sys.exit()
    print(train.describe(include="all"))
    #print(train.values.reshape())
    print(train.sort_values(by="Age", ascending=False))
    print(train.sort_values(by="Age", ascending=False)[:5])
    #print(test)




    #sys.exit()








