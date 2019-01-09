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
from petfinder.get_explore import read_data
from scipy.stats import ttest_rel
from petfinder.get_explore import Columns
from petfinder.preprocessing import conv_cat_variable
from petfinder.preprocessing import prepare_data
import matplotlib as mpl


def vis_cat_percentages(arr, indep_cols, dep_col):

    for col in indep_cols:
        fig, axes = plt.subplots(1, 2, figsize=[16, 9])
        df1 = arr[col].value_counts(normalize=True).rename("percentage").mul(100).reset_index()  # .sort_values(col)
        df1.rename(columns={"index":col}, inplace=True)
        ax1 = sns.catplot(x=col, y="percentage", data=df1, kind="bar", ax=axes[0])
        ax1.set_xticklabels(rotation=90)
        df2 = arr.groupby([col])[dep_col[0]].value_counts(normalize=True).rename('percentage').mul(
            100).reset_index()  # .sort_values(col)
        ax2 = sns.catplot(x=col,y="percentage", data=df2, hue=dep_col[0], kind="bar", ax=axes[1])
        ax2.set_xticklabels(rotation=90)
        plt.close(2)
        plt.close(3)
        plt.show()

def vis_relation(arr, indep_cols, dep_col):
    indep_cols.remove("RescuerID")
    for col in indep_cols:
        print(col)
        if col in Columns.ind_num_cat_columns.value :
            mpl.rcParams['axes.labelsize'] = 10
            mpl.rcParams['xtick.labelsize'] = 10
            mpl.rcParams['ytick.labelsize'] = 10
            f, axes = plt.subplots(2, 2, figsize=[16, 9])
            f.suptitle(col +' catplot')
            sns.catplot(x=col, y=dep_col[0], data=arr, kind="box", ax=axes[0,0])
            sns.catplot(x=col, y=dep_col[0], data = arr[arr["Type"] == 1], kind="box", ax=axes[1,0])
            sns.catplot(x=col, y=dep_col[0], data=arr[arr["Type"] == 2], kind="box", ax=axes[1,1])
            for i in range(2,5):
                plt.close(i)
        elif col in Columns.ind_cont_columns.value:
            mpl.rcParams['axes.labelsize'] = 5
            mpl.rcParams['xtick.labelsize'] = 5
            mpl.rcParams['ytick.labelsize'] = 5
            f, axes = plt.subplots(3, 6, figsize=[25, 25])
            f.suptitle(col + ' histogram')
            sns.distplot(arr[col], ax=axes[0, 0], axlabel="All")
            sns.distplot(arr[arr["AdoptionSpeed"] == 0][col], ax=axes[0, 1], axlabel="All")
            sns.distplot(arr[arr["AdoptionSpeed"] == 1][col], ax=axes[0, 2], axlabel="All")
            sns.distplot(arr[arr["AdoptionSpeed"] == 2][col], ax=axes[0, 3], axlabel="All")
            sns.distplot(arr[arr["AdoptionSpeed"] == 3][col], ax=axes[0, 4], axlabel="All")
            sns.distplot(arr[arr["AdoptionSpeed"] == 4][col], ax=axes[0, 5], axlabel="All")
            sns.distplot(arr[arr["Type"] == 1][col], ax=axes[1,0],axlabel="Dog")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 0) & (arr["Type"] == 1)][col], ax=axes[1, 1], axlabel="Dog for Adoption Speed 0")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 1) & (arr["Type"] == 1)][col], ax=axes[1, 2], axlabel="Dog for Adoption Speed 1")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 2) & (arr["Type"] == 1)][col], ax=axes[1, 3], axlabel="Dog for Adoption Speed 2")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 3) & (arr["Type"] == 1)][col], ax=axes[1, 4], axlabel="Dog for Adoption Speed 3")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 4) & (arr["Type"] == 1)][col], ax=axes[1, 5], axlabel="Dog for Adoption Speed 4")
            sns.distplot(arr[arr["Type"] == 2][col], ax=axes[2,0],axlabel="Cat")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 0) & (arr["Type"] == 2)][col], ax=axes[2, 1], axlabel="Cat for Adoption Speed 0")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 1) & (arr["Type"] == 2)][col], ax=axes[2, 2], axlabel="Cat for Adoption Speed 1")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 2) & (arr["Type"] == 2)][col], ax=axes[2, 3], axlabel="Cat for Adoption Speed 2")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 3) & (arr["Type"] == 2)][col], ax=axes[2, 4], axlabel="Cat for Adoption Speed 3")
            sns.distplot(arr[(arr["AdoptionSpeed"] == 4) & (arr["Type"] == 2)][col], ax=axes[2, 5], axlabel="Cat for Adoption Speed 4")

            for i in range(2,18):
                plt.close(i)

        plt.show()

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    train = conv_cat_variable(train)
    test = conv_cat_variable(test)
    # sns.catplot(y="Age", x="AdoptionSpeed", data = train, kind = "box")
    #sns.distplot(train.Color3)
    #f, axes = plt.subplots(5, 8, figsize=[16, 9])


    x_train, y_train, x_test, id_test = prepare_data()

    #vis_relation(pd.concat([x_train, y_train], axis = 1), Columns.ind_cont_columns.value+Columns.ind_num_cat_columns.value, Columns.dep_columns.value)
    vis_cat_percentages(train, Columns.ind_num_cat_columns.value, Columns.dep_columns.value)
    #conv_cat_variable(train)
    #print(train)
    #train.hist()
    #plt.show()