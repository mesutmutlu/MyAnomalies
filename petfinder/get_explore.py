import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import os
import json
from enum import Enum
from sklearn.feature_selection import VarianceThreshold
import time
import numpy as np



class Paths(Enum):
    if sys.platform == "linux":
        base = "/home/mesut/kaggle/petfinder.my/"
    else:
        base = "C:/datasets/petfinder.my/"

class Columns(Enum):
    ind_cont_columns = ["Age", "Fee", "VideoAmt", "PhotoAmt", "Quantity",
                        "FurLength", "MaturitySize", "DescScore", "DescMagnitude",
                        "DescLength", "NameLength"]
    ind_num_cat_columns = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "State",  "Pet_Breed"]
    ind_text_columns = ["Name", "Description"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]
    n_desc_svdcomp = 120
    desc_cols = ["desc_svd_" + str(i) for i in range(n_desc_svdcomp)]
    img_num_cols_1 = ["Vertex_X_1", "Vertex_Y_1", "Bound_Conf_1", "Bound_Imp_Frac_1",
                      "Dom_Blue_1", "Dom_Green_1", "Dom_Red_1",
                "RGBint_1", "Dom_Px_Fr_1", "Dom_Scr_1", "Lbl_Scr_1",]
    img_num_cols_2 = ["Vertex_X_2", "Vertex_Y_2", "Bound_Conf_2", "Bound_Imp_Frac_2",
                      "Dom_Blue_2", "Dom_Green_2", "Dom_Red_2",
                "RGBint_2", "Dom_Px_Fr_2", "Dom_Scr_2", "Lbl_Scr_2"]
    img_num_cols_3 = ["Vertex_X_3", "Vertex_Y_3", "Bound_Conf_3", "Bound_Imp_Frac_3",
                      "Dom_Blue_3", "Dom_Green_3", "Dom_Red_3",
                "RGBint_3", "Dom_Px_Fr_3", "Dom_Scr_3", "Lbl_Scr_3"]
    img_lbl_cols_1 = ["Lbl_Img_1"]
    img_lbl_cols_2 = ["Lbl_Img_2"]
    img_lbl_cols_3 = ["Lbl_Img_3"]
    img_lbl_col = ["Lbl_Img"]
    n_img_anl = 3
    n_iann_svdcomp = 3
    iann_cols = ["iann_svd_" + str(i) for i in range(n_iann_svdcomp)]
    ft_cat_cols = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                   "Vaccinated", "Dewormed", "Sterilized", "Health", "State",
                   "FurLength", "MaturitySize"]
    ft_new_cols = ["Age", "Fee", "VideoAmt", "PhotoAmt", "Quantity"]
    def feature_cols():
        tmp_ft_cols = []
        for cc in ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                       "Vaccinated", "Dewormed", "Sterilized", "Health", "State",
                       "FurLength", "MaturitySize"]:
            #print(cc)
            for a in ["SUM", "STD", "MAX", "SKEW", "MIN", "MEAN", "COUNT"]:
                #print(a)
                if a != "COUNT":
                    for x in ["Age", "Fee", "VideoAmt", "PhotoAmt", "Quantity", "AdoptionSpeed"]:
                        #print(x)
                        tmp_ft_cols.append(cc+"_"+a+"(Pets."+x+")")
                else:
                    tmp_ft_cols.append(cc+"_"+a+"(Pets)")
        return tmp_ft_cols

    ft_cols = feature_cols()
    item_type_incols = ["RescuerID", "Breed1", "Breed2", "Color1", "Color2", "Color3", "Pet_Breed"]
    item_type_cols = [c + "_Type" for c in item_type_incols]
    item_adp_cols = [c + "_Adp" for c in
                     ["RescuerID_Type", "Breed1_Type", "Breed2_Type", "Color1_Type", "Color2_Type", "Color3_Type"]]

    fee_mean_incols = ["Breed1", "Breed2", "Age", "Quantity"]
    fee_mean_cols = ["Fee_" + c for c in fee_mean_incols]


    @staticmethod
    def list(t):
        #return list(map(lambda c: c.value, filter(lambda x: x*2/6. != 1, range(5))))
        print([name for name, member in Columns.__members__.items() if member.name == t])
        print(Columns.value.__name__)

def read_data():

    train = pd.read_csv(Paths.base.value+"train/train.csv")
    test = pd.read_csv(Paths.base.value+"test/test.csv")
    breed_labels = pd.read_csv(Paths.base.value + "breed_labels.csv")

    breed_labels["Type_Breed"] = breed_labels.apply( lambda  x: (x["Type"], x["BreedID"]), axis=1)
    train["Type_Breed1"] = train.apply(lambda  x: (x["Type"], x["Breed1"]), axis=1)
    train["isin_Type_Breed1"] = train.apply(lambda x: True if x["Breed1"] == 0 else  x["Type_Breed1"] in breed_labels["Type_Breed"].values.tolist(), axis=1)
    train["Type_Breed2"] = train.apply(lambda  x: (x["Type"], x["Breed2"]), axis=1)
    train["isin_Type_Breed2"] = train.apply(lambda x: True if x["Breed2"] == 0 else x["Type_Breed2"] in breed_labels["Type_Breed"].values.tolist(), axis=1)

    train = train[train["isin_Type_Breed1"] == True]
    train = train[train["isin_Type_Breed2"] == True]
    train.drop(["Type_Breed1", "Type_Breed2", "isin_Type_Breed1", "isin_Type_Breed2"], axis=1, inplace=True)

    rc_snt = 0
    train_snt = get_desc_anly("train", rc_snt)
    test_snt = get_desc_anly("test", rc_snt)
    train = train.set_index("PetID").join(train_snt.set_index("PetID")).reset_index()
    test = test.set_index("PetID").join(test_snt.set_index("PetID")).reset_index()

    rc_img = 0
    train_metadata_1 = get_img_meta("train", "1", rc_img)
    train_metadata_2 = get_img_meta("train", "2", rc_img)
    train_metadata_3 = get_img_meta("train", "3", rc_img)
    test_metadata_1 = get_img_meta("test", "1", rc_img)
    test_metadata_2 = get_img_meta("test", "2", rc_img)
    test_metadata_3 = get_img_meta("test", "3", rc_img)

    train_metadata = train_metadata_1.set_index("PetID").join(train_metadata_2.set_index("PetID")).join(
        train_metadata_3.set_index("PetID")).reset_index()
    test_metadata = test_metadata_1.set_index("PetID").join(test_metadata_2.set_index("PetID")).join(
        test_metadata_3.set_index("PetID")).reset_index()

    train_metadata["Lbl_Img"] = train_metadata["Lbl_Img_1"] + train_metadata["Lbl_Img_2"] + train_metadata["Lbl_Img_3"]
    test_metadata["Lbl_Img"] = test_metadata["Lbl_Img_1"] + test_metadata["Lbl_Img_2"] + test_metadata["Lbl_Img_3"]

    train_metadata.drop(["Lbl_Img_1", "Lbl_Img_2", "Lbl_Img_3"], axis=1, inplace=True)
    test_metadata.drop(["Lbl_Img_1", "Lbl_Img_2", "Lbl_Img_3"], axis=1, inplace=True)

    train = train.set_index("PetID").join(train_metadata.set_index("PetID")).reset_index()
    test = test.set_index("PetID").join(test_metadata.set_index("PetID")).reset_index()


    return train, test



def get_desc_anly(type, recalc):
    if recalc == 1:
        if type == "train":
            path = Paths.base.value + "train_sentiment/"  # ../input/train_sentiment/
        elif type == "test":
            path = Paths.base.value + "test_sentiment/"  # ../input/test_sentiment/
        print("Getting description sentiment analysis for", type + "_sentiment.csv")
        files = [f for f in os.listdir(path) if (f.endswith('.json') & os.path.isfile(path + f))]

        df = pd.DataFrame(columns=["PetID", "DescScore", "DescMagnitude"])
        i = 0
        for f in files:
            # print(path + f)
            with open(path + f, encoding="utf8") as json_data:
                data = json.load(json_data)
            # print(data)
            # pf = pd.DataFrame.from_dict(data, orient='index').T.set_index('index')

            # print(data["documentSentiment"]["score"],data["documentSentiment"]["magnitude"])
            df.loc[i] = [f[:-5], data["documentSentiment"]["score"], data["documentSentiment"]["magnitude"]]
            i = i + 1
        df.to_csv(Paths.base.value + type + "_sentiment.csv", index=False)
    elif recalc == 0:
        df = pd.read_csv(Paths.base.value + type + "_sentiment.csv")
    return df


def get_img_meta(type, img_num, recalc):
    # getting image analyse metadata
    if recalc == 1:
        if type == "train":
            path = Paths.base.value+"train_metadata/"
        else:
            path = Paths.base.value+"test_metadata/"

        if img_num == "1":
            cols = Columns.iden_columns.value + Columns.img_num_cols_1.value + Columns.img_lbl_cols_1.value
            df_imgs = pd.DataFrame(columns=cols)
        elif img_num == "2":
            cols = Columns.iden_columns.value + Columns.img_num_cols_2.value + Columns.img_lbl_cols_2.value
            df_imgs = pd.DataFrame(columns=cols)
        elif img_num == "3":
            cols = Columns.iden_columns.value + Columns.img_num_cols_3.value + Columns.img_lbl_cols_3.value
            df_imgs = pd.DataFrame(columns=cols)
        else:
            print("This function supports images until 3rd, so img_num should be < = 3")
            return False

        print("Getting image analyse metadata for", type, img_num, "files")

        images = [f for f in sorted(os.listdir(path)) if
                  (f.endswith("-" + img_num + ".json") & os.path.isfile(path + f))]

        i = 0
        for img in images:
            PetID = img[:-7]
            # print(i, PetID,k, img, (img[-6:-5]), l_petid)

            with open(path + img, encoding="utf8") as json_data:
                data = json.load(json_data)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('x', -1)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('y', -1)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0].get('confidence', -1)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue', 255)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green', 255)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red', 255)
            RGBint = (dominant_red << 16) + (dominant_green << 8) + dominant_blue
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0].get('pixelFraction',
                                                                                                       -1)
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

            df_imgs.loc[i, cols] = [PetID, vertex_x, vertex_y, bounding_confidence, bounding_importance_frac, RGBint,
                                    dominant_blue, dominant_green, dominant_red,
                                    dominant_pixel_frac, dominant_score, label_score, label_description]

            i += 1

        #print(df_imgs.head())
        df_imgs.to_csv(Paths.base.value + type + "_metadata-" + img_num + ".csv", index=False)
    elif recalc == 0:
        df_imgs = pd.read_csv(Paths.base.value + type + "_metadata-" + img_num + ".csv")

    return df_imgs


def set_pet_breed(b1, b2):
    if (b1 in (0, 307)) & (b2 in (0, 307)):
        return 4
    elif (b1 == 307) & (b2 not in (0, 307)):
        return 3
    elif (b2 == 307) & (b1 not in (0, 307)):
        return 3
    elif (b1 not in (0, 307)) & (b2 not in (0, 307)) & (b1 != b2):
        return 2
    elif (b1 == 0) & (b2 not in (0, 307)):
        return 1
    elif (b2 == 0) & (b1 not in (0, 307)):
        return 1
    elif (b1 not in (0, 307)) & (b2 not in (0, 307)) & (b1 == b2):
        return 0
    else:
        return 4



if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #train, test = read_data()
    #print(train.corr())
    #print(sys.platform)

    train, test = read_data()

    X=train[["MaturitySize"]]
    y=train[["AdoptionSpeed"]]
    R = 10
    sums = np.zeros([len(X), 1])
    lens = np.zeros([len(X), 1])
    randomness =1+np.random.normal(scale = 0.03, size=len(sums)).reshape(len(sums), 1)
    for c in X.columns.values:
        for uval in X[c].unique():
            indexes = X[X[c] == uval].index
            s_tot = y.loc[indexes].sum()
            l = len(indexes)-1+R
            lens[indexes, 0] = 1/l
            #print(uval, indexes, s_tot)
            for i in indexes:
                s_ind = s_tot - int(y.loc[i])
                sums[i,0] = s_ind
        print(lens.shape)
        #print(lens)
        print(sums.shape)
        #print(sums)
        print(randomness.shape)
        print(randomness)
        print(sums*lens*randomness)





    sys.exit()

    train["Pet_Breed"] = train.apply(lambda x: set_pet_breed(x['Breed1'], x['Breed2']), axis=1)

    print(len(train))

    df = train.groupby("Pet_Breed").agg({'PetID': 'count', 'AdoptionSpeed': 'mean'})
    print(df)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(df)
    print(scaler.transform(df))




    #print(h.sort_values(by=['count', 'mean'],ascending=False))



    #sys.exit()








