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
    n_desc_svdcomp = 120
    desc_cols = ["desc_svd_" + str(i) for i in range(n_desc_svdcomp)]
    img_cols_1 = ["Vertex_X_1", "Vertex_Y_1", "Bound_Conf_1", "Bound_Imp_Frac_1",
                "RGBint_1", "Dom_Px_Fr_1", "Dom_Scr_1", "Lbl_Dsc_1", "Lbl_Scr_1",]
    img_cols_2 = ["Vertex_X_2", "Vertex_Y_2", "Bound_Conf_2", "Bound_Imp_Frac_2",
                "RGBint_2", "Dom_Px_Fr_2", "Dom_Scr_2", "Lbl_Dsc_2", "Lbl_Scr_2"]
    img_cols_3 = ["Vertex_X_3", "Vertex_Y_3", "Bound_Conf_3", "Bound_Imp_Frac_3",
                "RGBint_3", "Dom_Px_Fr_3", "Dom_Scr_3", "Lbl_Dsc_3", "Lbl_Scr_3"]
    img_ann_desc = ["Lbl_Dsc"]
    n_img_anl = 3
    n_iann_svdcomp = 10
    iann_cols = ["iann_svd_" + str(i) for i in range(n_iann_svdcomp)]




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

    calc_img = 0
    if os.path.exists(Paths.base.value+"train_metadata/train_metadata.csv"):
        train_img = pd.read_csv(Paths.base.value+"train_metadata/train_metadata.csv")
    else:
        train_img = get_img_meta("train", calc_img)
    if os.path.exists(Paths.base.value+"test_metadata/test_metadata.csv"):
        test_img = pd.read_csv(Paths.base.value+"test_metadata/test_metadata.csv")
    else:
        test_img = get_img_meta("test", calc_img)


    train = train.set_index("PetID").join(train_snt.set_index("PetID")).reset_index()
    test = test.set_index("PetID").join(test_snt.set_index("PetID")).reset_index()

    train = train.set_index("PetID").join(train_img.set_index("PetID")).reset_index()
    test = test.set_index("PetID").join(test_img.set_index("PetID")).reset_index()
    print(train.columns.values)
    train["Lbl_Dsc"].fillna("none",  inplace=True)
    train[Columns.img_cols_1.value.remove("Lbl_Dsc_1")].fillna("none", inplace=True)
    train[Columns.img_cols_2.value.remove("Lbl_Dsc_2")].fillna("none", inplace=True)
    train[Columns.img_cols_3.value.remove("Lbl_Dsc_3")].fillna("none", inplace=True)
    test["Lbl_Dsc"].fillna("none",  inplace=True)
    test[Columns.img_cols_1.value.remove("Lbl_Dsc_1")].fillna("none", inplace=True)
    test[Columns.img_cols_2.value.remove("Lbl_Dsc_2")].fillna("none", inplace=True)
    test[Columns.img_cols_3.value.remove("Lbl_Dsc_3")].fillna("none", inplace=True)


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

        files = [f for f in sorted(os.listdir(path)) if (f.endswith('.json') & os.path.isfile(path+f))]

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


def get_img_meta(type, rc):

    if type == "train":
        path = Paths.base.value + "train_metadata/"  # ../input/train_sentiment/
        fpath = Paths.base.value + "train_metadata/train_metadata.csv"
    else:
        path = Paths.base.value + "test_metadata/"  # ../input/test_sentiment/
        fpath = Paths.base.value + "test_metadata/test_metadata.csv"

        if rc == 1 or not (os.path.exists(fpath)):
            if os.path.exists(fpath):
                os.remove(fpath)

    df_imgs = pd.DataFrame(columns=Columns.iden_columns.value + Columns.img_cols_1.value + Columns.img_cols_2.value
                                   + Columns.img_cols_3.value + Columns.img_ann_desc.value)

    images = [f for f in sorted(os.listdir(path)) if ((f.endswith('-1.json') or f.endswith('-2.json') or f.endswith('-3.json')) & os.path.isfile(path + f))]

    i = 0
    l_petid = ""
    k = 0
    for img in images:
        PetID = img[:-7]
        if (l_petid != PetID) & (l_petid != ""):
            k += 1
        print(i, PetID,k, img, (img[-6:-5]), l_petid)

        with open(path + img, encoding="utf8") as json_data:
            data = json.load(json_data)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('x',-1)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('y',-1)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0].get('confidence',-1)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue', 255)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green', 255)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red', 255)
        RGBint = (dominant_red << 16) + (dominant_green << 8) + dominant_blue
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0].get('pixelFraction', -1)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0].get('score', -1)

        if data.get('labelAnnotations'):
            label_description = ""
            label_score = 0
            j = 1
            for ann in data.get('labelAnnotations'):
                if ann.get('score', 0) >= 0.80:
                    label_score = (ann.get('score', 0) + label_score) / j
                    label_description = label_description + " " + ann.get("description", "nothing")
                    j += 1
        else:
            label_description = 'nothing'
            label_score = -1

        if img[-6:-5] == "1":
            df_imgs.loc[k, Columns.iden_columns.value + Columns.img_cols_1.value] = [PetID,vertex_x, vertex_y, bounding_confidence,
                                                                                     bounding_importance_frac, RGBint,
                                                                                    dominant_pixel_frac, dominant_score, label_description,label_score]

        if img[-6:-5] == "2":
            df_imgs.loc[k, Columns.img_cols_2.value] = [vertex_x, vertex_y, bounding_confidence,
                                                        bounding_importance_frac, RGBint,
                                                        dominant_pixel_frac, dominant_score, label_description,
                                                        label_score]

        if img[-6:-5] == "3":
            df_imgs.loc[k, Columns.img_cols_3.value] = [vertex_x, vertex_y, bounding_confidence,
                                                        bounding_importance_frac, RGBint,
                                                        dominant_pixel_frac, dominant_score, label_description,
                                                            label_score]


        if (i == 1):
            l_petid = PetID

        if (l_petid != PetID):
            l_petid = PetID

        i += 1

    df_imgs["Lbl_Dsc"] = df_imgs["Lbl_Dsc_1"] + " " + df_imgs["Lbl_Dsc_2"] + " " + df_imgs["Lbl_Dsc_3"]
    df_imgs.drop(["Lbl_Dsc_1","Lbl_Dsc_2", "Lbl_Dsc_3"], axis=1, inplace=True)

    df_imgs["Lbl_Dsc"].fillna("none", inplace=True)
    df_imgs.fillna(-1, inplace=True)

    if os.path.isfile(fpath):
        pd.read_csv(fpath)

    df_imgs.to_csv(fpath, index=False)

    return df_imgs


if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #train, test = read_data()
    #print(train.corr())
    #print(sys.platform)
    train, test = read_data()

    print(train.head())
    sys.exit()
    print(train.describe(include="all"))
    #print(train.values.reshape())
    print(train.sort_values(by="Age", ascending=False))
    print(train.sort_values(by="Age", ascending=False)[:5])
    #print(test)




    #sys.exit()








