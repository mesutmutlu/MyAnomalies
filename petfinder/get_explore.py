import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import os
import json
from enum import Enum
import datetime
from sklearn.feature_selection import VarianceThreshold
import time
import numpy as np
import scipy as sp
import glob
from petfinder.ownestimator import RatioOrdinalClassfier
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb


class Paths(Enum):
    if sys.platform == "linux":
        base = "/home/mesut/kaggle/petfinder.my/"
    else:
        base = "C:/datasets/petfinder.my/"



class FileNum(Enum):
    train_image_files = sorted(glob.glob(Paths.base.value+"train_images/*.jpg"))
    train_metadata_files = sorted(glob.glob(Paths.base.value+"train_metadata/*.json"))
    train_sentiment_files = sorted(glob.glob(Paths.base.value+"train_sentiment/*.json"))

    test_image_files = sorted(glob.glob(Paths.base.value + "test_images/*.jpg"))
    test_metadata_files = sorted(glob.glob(Paths.base.value + "test_metadata/*.json"))
    test_sentiment_files = sorted(glob.glob(Paths.base.value + "test_sentiment/*.json"))

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


def get_all_img_meta(type, recalc):
    # getting image analyse metadata
    if recalc == 1:
        if type == "train":
            path = Paths.base.value+"train_metadata/"
        else:
            path = Paths.base.value+"test_metadata/"

        all_images = [f for f in sorted(os.listdir(path)) if
                  (f.endswith(".json") & os.path.isfile(path + f))]
        len_images = len(all_images)

        pets = list(set([f[:f.find("-")] for f in sorted(os.listdir(path)) if
                  (f.endswith(".json") & os.path.isfile(path + f))]))

        np_pets = np.asarray(pets).reshape(len(pets), 1)
        df_cols = ["P_RGB", "P_Dom_Px_Frac", "P_Dom_Score", "P_Vertex_X", "PVertex_Y",
                 "P_Bound_Conf", "P_Bound_Imp_Frac", "P_Label_Score", "P_Label_Description"]
        np_data = np.zeros((len(pets),len(df_cols)))
        data = np.concatenate((np_pets,np_data), axis=1 )
        df_pet_img_meta = pd.DataFrame(columns=["PetID"]+df_cols, data=data)
        df_pet_img_meta["P_Label_Description"] = ""
        df_pet_img_meta.set_index("PetID", inplace=True)
        #print(df_pet_img_meta.head())

        h = 1
        for pet in pets:
            images = [k for k in all_images if pet in k]

            p_rgb = 0
            p_dominant_pixel_frac = 0
            p_dominant_score = 0
            p_vertex_x = 0
            p_vertex_y = 0
            p_bounding_confidence = 0
            p_bounding_importance_frac = 0
            p_label_score = 0
            p_label_description = ""
            num_of_images = len(images)



            for img in images:
                imgnum = img.split("-", 1)[1].strip(".json")
                with open(path + img, encoding="utf8") as json_data:
                    image = json.load(json_data)

                print(img, h, len_images)
                h = h+1

                i_rgb = 0
                i2_rgb = 0
                i_dominant_pixel_frac = 0
                i_dominant_score = 0
                i_label_score = 0
                i_label_description = ""
                i_num_colors = 0
                i_num_label_annotations = 0

                i_vertex_x = image['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('x', 0)
                i_vertex_y = image['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('y', 0)
                i_bounding_confidence = image['cropHintsAnnotation']['cropHints'][0].get('confidence', 0)
                i_bounding_importance_frac = image['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', 0)

                if image.get('imagePropertiesAnnotation').get("dominantColors").get("colors"):
                    i_num_colors = len(image.get('imagePropertiesAnnotation').get("dominantColors").get("colors"))
                    t_rgb = 0
                    for color in image.get('imagePropertiesAnnotation').get("dominantColors").get("colors"):
                        #print(color, color.get("color").get("red"))
                        r = color.get("color").get('red', 255)
                        g = color.get("color").get('green', 255)
                        b = color.get("color").get('blue', 255)
                        #rgbint = ((r << 16) + (g << 8) + b)
                        m_rgb = (r**2 + g**2 + b**2)/3
                        t_rgb = t_rgb + m_rgb
                        i_dominant_pixel_frac = i_dominant_pixel_frac + color.get('pixelFraction', 0)
                        i_dominant_score = i_dominant_score + color.get('score', 0)

                    if i_num_colors>0:
                        if t_rgb > 0:
                            i_rgb = np.sqrt(t_rgb/i_num_colors)
                        if i_dominant_score > 0:
                            i_dominant_score = i_dominant_score/i_num_colors
                    #print(i_rgb, i_dominant_pixel_frac, i_dominant_score)
                if image.get('labelAnnotations'):
                    i_num_label_annotations = len(image.get('labelAnnotations'))
                    for ann in image.get('labelAnnotations'):
                        i_label_score = ann.get('score', 0) + i_label_score
                        if ann.get("description"):
                            i_label_description = i_label_description + " " + ann.get("description", "")

                    if i_num_label_annotations>0:
                        if i_label_score > 0:
                            i_label_score = i_label_score/i_num_label_annotations

                    #print(i_label_score, i_label_description)

                p_rgb = p_rgb + i_rgb**2
                p_dominant_pixel_frac = p_dominant_pixel_frac + i_dominant_pixel_frac
                p_dominant_score = p_dominant_score + i_dominant_score
                p_vertex_x = p_vertex_x + i_vertex_x
                p_vertex_y = p_vertex_y + i_vertex_y
                p_bounding_confidence = p_bounding_confidence + i_bounding_confidence
                p_bounding_importance_frac = p_bounding_importance_frac + i_bounding_importance_frac
                p_label_score = p_label_score + i_label_score
                p_label_description = p_label_description + " " + i_label_description

            p_rgb = np.sqrt(p_rgb/num_of_images)
            p_dominant_pixel_frac = p_dominant_pixel_frac/num_of_images
            p_dominant_score = p_dominant_score/num_of_images
            p_vertex_x = p_vertex_x/num_of_images
            p_vertex_y = p_vertex_y/num_of_images
            p_bounding_confidence = p_bounding_confidence/num_of_images
            p_bounding_importance_frac = p_bounding_importance_frac/num_of_images
            p_label_score = p_label_score/num_of_images


            df_pet_img_meta.loc[pet, df_cols] = [p_rgb, p_dominant_pixel_frac, p_dominant_score, p_vertex_x, p_vertex_y,
                                                 p_bounding_confidence, p_bounding_importance_frac, p_label_score,
                                                 p_label_description]
            #print(df_pet_img_meta.head())
        print(df_pet_img_meta.head())
        df_pet_img_meta.reset_index().to_csv(Paths.base.value + type + "_metadata_all.csv", index=False)
    elif recalc == 0:
        df_pet_img_meta = pd.read_csv(Paths.base.value + type + "_metadata_all.csv")

    return df_pet_img_meta

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

def stdType(min, max, mean, std, value):
    if min <= value < mean - 5 * std:
        return 0
    elif  mean -5*std <= value < mean - 4*std:
        return 1
    elif mean -4*std <= value < mean - 3*std:
        return 2
    elif mean - 3*std <= value < mean - 2*std:
        return 3
    elif mean - 2*std <= value < mean - std:
        return 4
    elif mean - std <= value < mean:
        return 5
    elif mean <= value < mean + std:
        return 6
    elif mean + std <= value < mean + 2*std:
        return 7
    elif mean + 2*std <= value < mean + 3*std:
        return 8
    elif mean + 3*std <= value < mean + 4*std:
        return 9
    elif mean + 4*std <= value < mean + 5*std:
        return 10
    elif mean + 5*std <= value <= max:
        return 11





if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #train, test = read_data()
    #print(train.corr())
    #print(sys.platform)
    #get_all_img_meta("train", 1)
    start = datetime.datetime.now()
    from joblib import Parallel, delayed

    cat_cols = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]

    from imblearn.over_sampling import SMOTE
    import random
    import string

    random_resc = []
    for i in range (100):
        rnd = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
        random_resc.append(rnd)

    print(np.asarray(random_resc).reshape(-1,1))


    train = pd.read_csv(Paths.base.value + "train/train.csv")

    print(train["PetID"], len(train["PetID"][1500]))

    print(len(train), len(train["RescuerID"].unique()))
    sys.exit()



    train.drop(["Name", "Description", "PetID", "RescuerID"], axis=1, inplace=True)
    sm = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=10)
    X_res, y_res = sm.fit_resample(train.drop("AdoptionSpeed", axis=1), train["AdoptionSpeed"])
    print(X_res.shape, y_res.reshape(-1,1).shape)
    train_res =pd.DataFrame(data = np.concatenate([X_res, y_res.reshape(-1,1)], axis=1), columns= train.columns.values.tolist())

    print(train_res.head())
    tr_f = pd.concat([train, train_res], axis=0)

    tr_f["AdoptionSpeed"].hist()
    plt.show()
    sys.exit()
    x_train = train.drop("AdoptionSpeed", axis=1)
    y_train = train["AdoptionSpeed"]





    from keras.datasets import reuters
    from keras import models, layers
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.utils.np_utils import to_categorical

    y_train = to_categorical(np.array(y_train.values))
    #from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    x_train['State'] = le.fit_transform(x_train['State'])

    #enc = OneHotEncoder(handle_unknown='ignore')
    cat = np.array([])
    for c in cat_cols:
        print(datetime.datetime.now(), c)
        t_cat = to_categorical(x_train[c])
        if cat.size == 0:
            cat = t_cat
        else:
            cat = np.concatenate((cat, t_cat), axis=1)

        x_train.drop(c, inplace=True, axis=1)
        print(cat.shape)

    print(datetime.datetime.now(), "x_train to ndarray")
    x_train = np.array(x_train.values)

    print(datetime.datetime.now(), "concatenate xtran and cat")
    x_train = np.concatenate((x_train, cat), axis=1)

    print(datetime.datetime.now(), x_train.shape, y_train.shape)



    from sklearn.model_selection import train_test_split

    partial_x_train, x_val, partial_y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

    print(x_train)
    model = models.Sequential()
    model.add(layers.Dense(1000, activation='relu', input_shape=(partial_x_train.shape[1],)))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if 1 == 1:
        history = model.fit(partial_x_train, partial_y_train, epochs=20,
                            batch_size=512, validation_data=(x_val, y_val))

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(loss)+1)

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.show()

        model.evaluate(x_test, )

    sys.exit()


    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    tfv = TfidfVectorizer(min_df=4,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')

    # Fit TFIDF
    train["Description"].fillna("", inplace=True)
    tfv.fit(list(train["Description"]))
    X = tfv.transform(train["Description"])
    print(len(tfv.vocabulary_))
    #df = pd.DataFrame(data=X.toarray())
    #print(df )


    sys.exit()

    from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer

    cols = ["Age", "Fee", "PhotoAmt", "VideoAmt", "Quantity"]

    norm = np.random.normal(0, 0.1, 1000)
    from scipy.stats import skewtest, normaltest

    print(normaltest(norm))
    rng = np.random.RandomState(304)
    qt = QuantileTransformer(output_distribution='normal', random_state=rng)
    pt = PowerTransformer(method="yeo-johnson")



    for c in cols:
        f, axes = plt.subplots(2, 2)
        axes[0, 0].hist(train[c], bins='auto')
        axes[0, 0].set_title(c + " notransform:"+str(normaltest(train[c])[1]))

        qt_t = qt.fit_transform(train[c].values.reshape(-1,1))
        axes[0, 1].hist(qt_t, bins='auto', label=str(normaltest(qt_t)[1]))
        axes[0, 1].set_title("quantiletransform:"+str(normaltest(qt_t)[1]))

        pt_t = pt.fit_transform(train[c].values.reshape(-1,1))
        axes[1, 0].hist(pt_t, bins='auto', label=str(normaltest(pt_t)[1]))
        axes[1, 0].set_title("powertransform:"+str(normaltest(pt_t)[1]))

        log_t = np.log(train[c]+1)
        axes[1, 1].hist(log_t.dropna().values, bins='auto', label=  str(normaltest(log_t)))
        axes[1, 1].set_title("logtransform:"+str(normaltest(log_t)[1]))
        plt.show()





    sys.exit()


    from sklearn.feature_selection import  chi2

    chir = chi2(train[cat_cols], train["AdoptionSpeed"])
    chi_data = np.concatenate((chir[0].reshape(len(chir[0]), 1), chir[1].reshape(len(chir[0]), 1)), axis=1)
    df_chi = pd.DataFrame(index = cat_cols, data= chi_data, columns=["chi2", "pval"])
    df_chi["Select"] = df_chi["pval"] <0.01
    print(df_chi)