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
from petfinder.getdata import read_data
from scipy.stats import ttest_rel


def check_percentages(arr, indep_cols, dep_col):

    for col in indep_cols:
        #fig, axes = plt.subplots(1, 2)
        df1 = arr[col].value_counts(normalize=True).rename("percentage").mul(100).reset_index()  # .sort_values(col)
        df1.rename(columns={"index":col}, inplace=True)
        ax1 = sns.catplot(x=col, y="percentage", data=df1, kind="bar")
        df2 = arr.groupby([col])[dep_col[0]].value_counts(normalize=True).rename('percentage').mul(
            100).reset_index()  # .sort_values(col)
        ax2 = sns.catplot(x=col,y="percentage", data=df2, hue=dep_col[0], kind="bar")
        plt.show()

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()

    ind_cont_columns = ["Age", "Fee", "VideoAmt", "PhotoAmt"]
    ind_num_cat_columns = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize",
                           "FurLength",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "State"]
    ind_cat_conv_columns = ["RescuerID"]
    ind_text_columns = ["Name", "Description"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]

    check_percentages(train[ind_num_cat_columns+dep_columns], ind_num_cat_columns, dep_columns)