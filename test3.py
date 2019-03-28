import pandas as pd
from enum import Enum
import sys
import matplotlib.pyplot as plt
import seaborn as sns

class Columns(Enum):
    rescuer_id = ["RescuerID"]
    ind_cont_columns = ["Age", "Fee", "VideoAmt", "PhotoAmt", "Quantity",
                        "DescScore", "DescMagnitude", "DescLength", "SentMagnitude", "SentMagnitute_Mean",
                        "SentScore", "SentScore_Mean", "EntSalience", "EntSalience_Mean",
                        "NameLength", "Pet_Maturity", "Color_Type", "Fee_Per_Pet"]
    ind_num_cat_columns = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "State", "MaturitySize",
                           "FurLength", "Pet_Breed", "Breed_Merge", "Pet_Purity", "Overall_Status"]
    ind_text_columns = ["Name", "Description", "Entities"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]
    n_desc_svd_comp = 120
    desc_svd_cols = ["desc_svd_" + str(i) for i in range(n_desc_svd_comp)]
    n_desc_nmf_comp = 5
    desc_nmf_cols = ["desc_nmf_" + str(i) for i in range(n_desc_nmf_comp)]
    img_num_cols_all = ["P_RGB", "P_Dom_Px_Frac", "P_Dom_Px_Frac_Mean", "P_Dom_Score", "P_Dom_Score_Mean", "P_Vertex_X",
                        "P_Vertex_Y",
                        "P_Bound_Conf", "P_Bound_Conf_Mean", "P_Bound_Imp_Frac", "P_Bound_Imp_Frac_Mean",
                        "P_Label_Score", "P_Label_Score_Mean"]
    img_num_cols_1 = ["Vertex_X_1", "Vertex_Y_1", "Bound_Conf_1", "Bound_Imp_Frac_1",
                      "Dom_Blue_1", "Dom_Green_1", "Dom_Red_1",
                      "RGBint_1", "Dom_Px_Fr_1", "Dom_Scr_1", "Lbl_Scr_1", ]
    img_num_cols_2 = ["Vertex_X_2", "Vertex_Y_2", "Bound_Conf_2", "Bound_Imp_Frac_2",
                      "Dom_Blue_2", "Dom_Green_2", "Dom_Red_2",
                      "RGBint_2", "Dom_Px_Fr_2", "Dom_Scr_2", "Lbl_Scr_2"]
    img_num_cols_3 = ["Vertex_X_3", "Vertex_Y_3", "Bound_Conf_3", "Bound_Imp_Frac_3",
                      "Dom_Blue_3", "Dom_Green_3", "Dom_Red_3",
                      "RGBint_3", "Dom_Px_Fr_3", "Dom_Scr_3", "Lbl_Scr_3"]
    img_lbl_cols_1 = ["Lbl_Img_1"]
    img_lbl_cols_2 = ["Lbl_Img_2"]
    img_lbl_cols_3 = ["Lbl_Img_3"]
    img_lbl_col = ["Lbl_Dsc"]
    n_iann_svd_comp = 5
    iann_svd_cols = ["iann_svd_" + str(i) for i in range(n_iann_svd_comp)]
    n_iann_nmf_comp = 5
    iann_nmf_cols = ["iann_nmf_" + str(i) for i in range(n_iann_nmf_comp)]
    n_entities_svd_comp = 5
    entities_svd_cols = ["entities_svd_" + str(i) for i in range(n_entities_svd_comp)]
    n_entities_nmf_comp = 5
    entities_nmf_cols = ["entities_nmf_" + str(i) for i in range(n_entities_nmf_comp)]
    item_cnt_incols = ["RescuerID", "Breed1", "Breed2", "Breed_Merge",
                       "Age"]  # , "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    item_cnt_cols = [c + "_Cnt" for c in item_cnt_incols]
    item_cnt_mtype_cols = [c + "_Cnt_MType" for c in item_cnt_incols]
    item_type_incols = ["RescuerID_Cnt", "Breed1_Cnt", "Breed2_Cnt", "Breed_Merge_Cnt",
                        "Age_Cnt"]  # , "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    item_type_cols = [c + "_StdType" for c in item_type_incols]
    item_adp_incols = ["RescuerID_Cnt_StdType",
                       "Breed_Merge_Cnt_StdType"]  # , "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    item_adp_cols = [c + "_Adp" for c in item_adp_incols]
    fee_mean_incols = ["Breed1", "Breed2", "Age", "Breed_Merge",
                       "State"] + item_cnt_cols + item_cnt_mtype_cols + item_type_cols
    fee_mean_cols = ["Fee_Per_Pet_" + c for c in fee_mean_incols]
    loo_incols = ["Pet_Breed", "State"]
    loo_cols = [c + "_Loo" for c in loo_incols]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    scaling_cols = ["Age", "Fee", "RescuerID_Cnt", "Breed1_Cnt", "Breed2_Cnt", "Breed_Merge_Cnt", "Age_Cnt",
                    "Fee_Per_Pet"] + fee_mean_cols
    kbin_incols = ["Age", "Fee",
                   "Fee_Per_Pet"] + item_cnt_cols  # , "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    kbin_cols = [c + "_Kbin" for c in kbin_incols]
    ft_cat_cols = ["Breed1", "Breed2", "Breed_Merge", "Overall_Status"] + item_cnt_cols + item_type_cols + kbin_cols
    ft_new_cols = ["Age", "Fee", "Quantity", "VideoAmt", "PhotoAmt"]
    agg_calc = ["STD", "SKEW", "MEAN", "MAX", "MIN"]

    def feature_cols(fcc, fnc, agg_calc):
        tmp_ft_cols = []
        for cc in fcc:
            # print(cc)
            for a in agg_calc:
                # print(a)
                if a != "COUNT":
                    for x in fnc:
                        # print(x)
                        tmp_ft_cols.append(cc + "_" + a + "(Pets." + x + ")")
                else:
                    tmp_ft_cols.append(cc + "_" + a + "(Pets)")
        return tmp_ft_cols

    ft_cols = feature_cols(ft_cat_cols, ft_new_cols, agg_calc)
    # "Type", "Gender", "Color1", "Color2", "Color3","Vaccinated", "Dewormed", "Sterilized", "FurLength",

    barplot_cols = ind_num_cat_columns + item_type_cols + kbin_cols + item_cnt_mtype_cols
    boxplot_cols = ind_cont_columns + desc_svd_cols + desc_nmf_cols + img_num_cols_all + img_num_cols_1 + img_num_cols_2 + img_num_cols_3 + img_lbl_cols_1 + img_lbl_cols_2 + img_lbl_cols_3 + iann_svd_cols + iann_nmf_cols + entities_svd_cols + entities_nmf_cols + ft_cols + item_cnt_cols + item_adp_cols + fee_mean_cols

print(1)
train = pd.read_csv("C:/Users/dtmemutlu/Downloads/train.csv")
#test = pd.read_csv("C:/Users/dtmemutlu/Downloads/test.csv")
cols = []
print(2)
x_train = train.drop(["AdoptionSpeed"], axis=1)
y_train = train[["AdoptionSpeed"]]

dogs = train[train["Type"]==1]
x_train_dogs = dogs.drop(["AdoptionSpeed"], axis=1)
y_train_dogs = dogs[["AdoptionSpeed"]]
cats = train[train["Type"]==2]
x_train_cats = cats.drop(["AdoptionSpeed"], axis=1)
y_train_cats = cats[["AdoptionSpeed"]]

import traceback
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr, SelectFwe

if 1 == 1:
    cols = []

    for c in Columns.boxplot_cols.value + Columns.item_type_cols.value + Columns.kbin_cols.value + Columns.item_cnt_mtype_cols.value:
        if c in x_train.columns.values.tolist():
            cols.append(c)

    i=1
    for c in Columns.ft_cols.value:
        if c not in x_train.columns.values.tolist():
            print(i, c)
            i+=1


    corr = pd.concat([x_train[cols], y_train], axis=1)
    # print(corr)
    print(3)
    corr_dogs = pd.concat([x_train_dogs[cols], y_train_dogs], axis=1)
    corr_cats = pd.concat([x_train_cats[cols], y_train_cats], axis=1)
    corr_all = pd.concat([corr.corr('spearman').loc["AdoptionSpeed", :], corr_dogs.corr('spearman').loc["AdoptionSpeed", :],
                          corr_cats.corr('spearman').loc["AdoptionSpeed", :]], axis=1)
    df_corr_all = pd.DataFrame(columns=(["All", "Dogs", "Cats"]), data=(corr_all.values), index=corr_all.index.values)
    plt.rcParams["figure.figsize"] = [10, 40]
    ax = plt.axes()
    #sns.heatmap(df_corr_all.sort_values(by=['All']), annot=True, fmt='.4f', ax=ax)
    #ax.set_title('Dataset correlation between independent continous/ordinal and dependent ordinal variables')
    #plt.show()

    df_corr_all.sort_values(by=["All"], ascending=False).to_csv("spearman_for_continous.csv")

    try:
        result = chi2(x_train[Columns.ind_num_cat_columns.value], y_train["AdoptionSpeed"].values.reshape(-1, 1))
        print(result)

        chi2_pvals = []
        for x in result[1]:
            chi2_pvals.append(x)
        print(chi2_pvals)
        df_chi2_pvals = pd.DataFrame(index=Columns.ind_num_cat_columns.value, columns=["AdoptionSpeed"],
                                     data=chi2_pvals).sort_values(by=["AdoptionSpeed"], ascending=False)
        df_chi2_pvals.to_csv("chitest_for_categorical.csv")
    except Exception as e:
        print(e)
        traceback.print_exc()

#sys.exit()

if 1 == 1:
    x_train_fs = x_train.copy()

    mic = mutual_info_classif(x_train_fs.astype("float"), y_train["AdoptionSpeed"].values.ravel(), random_state=42)
    df_mic = pd.DataFrame(data=mic, index=x_train.columns.values.tolist(), columns=["importance"])
    print(df_mic)
    print(df_mic.sort_values(by=["importance"]))
    df_mic.sort_values(by=["importance"]).to_csv("df_mic.csv")