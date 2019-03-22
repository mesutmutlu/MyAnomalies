import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
import numpy as np
import matplotlib.pyplot as plt
import sys


train_df = pd.read_csv("C:/Users/dtmemutlu/Downloads/train.csv")

print(train_df[1:])
sys.exit()
test_df =  pd.read_csv("C:/Users/dtmemutlu/Downloads/test.csv")
x_train = train_df.drop("AdoptionSpeed", axis=1)
y_train = train_df[["AdoptionSpeed"]]
x_test = test_df.drop("PetID", axis=1)
id_test = test_df[["PetID"]]

if 1 == 1:

    c_features = [train_df.columns.get_loc(c) for c in train_df.columns if c in
                  ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "State", "MaturitySize",
                           "FurLength", "Pet_Breed", "Breed_Merge"]]
    import random
    import string
    from imblearn.over_sampling import RandomOverSampler, SMOTENC
    print(len(train_df))
    print("Generating mock data with smote")
    ros = SMOTENC(random_state=0, sampling_strategy='minority', categorical_features=c_features)
    train_df_c = train_df[:1000]
    x_train_df_c = train_df_c.drop(["AdoptionSpeed"], axis=1)
    y_train_df_c = train_df_c["AdoptionSpeed"]
    x_res, y_res = ros.fit_resample(x_train_df_c, y_train_df_c)
    x_res = pd.DataFrame(data=x_res, columns=x_train_df_c.columns.values.tolist())
    y_res = pd.DataFrame(data=y_res, columns=["AdoptionSpeed"])
    print("Generated mock data with smote")
    pd.concat([x_res,y_res], axis=1).to_csv("./output/x_res.csv")
    train_df_c.to_csv("./output/train_df_c.csv")
    rnd_resc_arr = []
    sys.exit()

    x_res_diff = []
    i = 0
    l = len(x_res)
    for a in pd.concat([x_res, y_res], axis=1)[15000:].values:
        #print("processing", a)
        a = a.reshape(1, a.shape[0])
        a = a[0]
        if a in train_df_c.values:
            print("not", i, "/", l, a.shape, train_df_c.shape)
        elif a not in train_df_c.values:
            print("in", i)
            if x_res_diff == []:
                print("initialize array")
                print("new a:", a)
                x_res_diff = a
            else:
                print("array exist")
                print("new a:", a)
                x_res_diff = np.append(x_res_diff, a, axis=0)

        i += 1
    print(x_res_diff.shape)

    sys.exit()

    for i in range (int(len(x_res)/4)):
        rnd_RescuerID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
        rnd_resc_arr.append(rnd_RescuerID)
    x_res["RescuerID"] = x_res.apply(lambda x : random.choice(rnd_resc_arr), axis=1)

    rnd_petID_arr = []
    for i in range (len(x_res)):
        rnd_petID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(9)])
        rnd_petID_arr.append(rnd_petID)
    x_res["PetID"] = np.asarray(rnd_petID_arr).reshape(-1,1)

    train_df_smote = pd.concat([x_res, y_res], axis=1)
    print(len(train_df_smote))

    #print(x_res.shape, y_res.reshape(-1,1).shape)
    train_df = pd.concat([train_df, train_df_smote], axis=0, sort=False)
    print(len(train_df))
print("Done")