import math
import pandas as pd

df = pd.DataFrame(columns=['AdoptionSpeed'], data=[1.5, 2, 3.5])

print(df.a)

print(math.floor(3/2))

import sys
sys.exit()

from enum import Enum


class Columns(Enum):
    rescuer_id = ["RescuerID"]
    ind_cont_columns = ["Age", "Fee", "VideoAmt", "PhotoAmt", "Quantity",
                        "DescScore", "DescMagnitude", "DescLength", "SentMagnitude", "SentMagnitute_Mean",
                        "SentScore", "SentScore_Mean", "EntSalience", "EntSalience_Mean",
                        "NameLength", "Age_Type", "Color_Type", "Fee_Per_Pet"]
    ind_num_cat_columns = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "State", "MaturitySize",
                           "FurLength", "Pet_Breed", "Breed_Merge"]
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
    ft_cat_cols = ["Breed1", "Breed2", "Breed_Merge", "RescuerID_Type", "Sterilized"]
    ft_new_cols = ["Age", "Fee", "Quantity"]

    def feature_cols(fcc, fnc):
        tmp_ft_cols = []
        for cc in fcc:
            # print(cc)
            for a in ["STD"]:
                # print(a)
                if a != "COUNT":
                    for x in fnc:
                        # print(x)
                        tmp_ft_cols.append(cc + "_" + a + "(Pets." + x + ")")
                else:
                    tmp_ft_cols.append(cc + "_" + a + "(Pets)")
        return tmp_ft_cols

    ft_cols = feature_cols(ft_cat_cols, ft_new_cols)
    # "Type", "Gender", "Color1", "Color2", "Color3","Vaccinated", "Dewormed", "Sterilized", "FurLength",
    item_type_incols = ["RescuerID", "Breed1", "Breed2",
                        "Breed_Merge"]  # , "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    item_type_cols = [c + "_Type" for c in item_type_incols]
    item_adp_incols = ["RescuerID_Type"]  # , "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    item_adp_cols = [c + "_Adp" for c in item_adp_incols]
    fee_mean_incols = ["Breed1", "Breed2", "Age", "Breed_Merge"]
    fee_mean_cols = ["Fee_" + c for c in fee_mean_incols]
    loo_incols = ["Pet_Breed", "State"]
    loo_cols = [c + "_Loo" for c in loo_incols]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    scaling_cols = ["Age", "Fee", "VideoAmt", "PhotoAmt", "Quantity", "NameLength", "Fee_Per_Pet",
                    "DescLength"] + item_adp_cols + fee_mean_cols
    # item_type_cols +

print(Columns.ft_cols.value)


