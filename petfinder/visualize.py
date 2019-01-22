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
from petfinder.preprocessing import prepare_data
import matplotlib as mpl
from pylab import rcParams


def check_cat_perc(arr, cols):
    for col in cols:
        print(col)
        fig, axes = plt.subplots(2, 2, figsize=[16, 9])
        # axes[0,0].set_title("Perc. of " + col + " dist.")
        # axes[0,1].set_title("Perc. of " + col + " dist. by AdopSp.")
        # axes[1,0].set_title("Perc. of " + col + " dist. by AdopSp. for Dogs")
        # axes[1,1].set_title("Perc. of " + col + " dist. by AdopSp. for Cats")
        df1 = arr[col].value_counts(normalize=True).rename("percentage").mul(100).reset_index()  # .sort_values(col)
        df1.rename(columns={"index": col}, inplace=True)
        ax1 = sns.catplot(x=col, y="percentage", data=df1, kind="bar", ax=axes[0, 0])
        ax1.set_axis_labels("All "+col, "Percentage")
        ax1.set_xticklabels(rotation=90)
        df2 = arr.groupby([col])["AdoptionSpeed"].value_counts(normalize=True).rename('percentage').mul(100).reset_index()
        ax2 = sns.catplot(x=col, y="percentage", data=df2, hue="AdoptionSpeed", kind="bar", ax=axes[0, 1])
        ax2.set_axis_labels("All "+col, "Percentage")
        ax2.set_xticklabels(rotation=90)
        df2_d = arr[arr["Type"] == 1].groupby([col])["AdoptionSpeed"].value_counts(normalize=True).rename('percentage').mul(
            100).reset_index()
        ax2_d = sns.catplot(x=col, y="percentage", data=df2_d, hue="AdoptionSpeed", kind="bar", ax=axes[1, 0])
        ax2_d.set_axis_labels("Dogs for "+col, "Percentage")
        ax2_d.set_xticklabels(rotation=90)
        df2_c = arr[arr["Type"] == 2].groupby([col])["AdoptionSpeed"].value_counts(normalize=True).rename('percentage').mul(
            100).reset_index()
        ax2_c = sns.catplot(x=col, y="percentage", data=df2_c, hue="AdoptionSpeed", kind="bar", ax=axes[1, 1])
        ax2_c.set_axis_labels("Cats for "+col, "Percentage")
        ax2_c.set_xticklabels(rotation=90)
        plt.close(2)
        plt.close(3)
        plt.close(4)
        plt.close(5)
        plt.show()

def check_num_dist(arr, cols):
    for col in cols:
        print(col)
        mpl.rcParams['axes.labelsize'] = 5
        mpl.rcParams['xtick.labelsize'] = 5
        mpl.rcParams['ytick.labelsize'] = 5
        f, axes = plt.subplots(3, 6, figsize=[25, 25])
        f.suptitle(col + ' histogram')
        sns.distplot(arr[col], ax=axes[0, 0], axlabel="All")
        sns.distplot(arr[arr["AdoptionSpeed"] == 0][col], ax=axes[0, 1], axlabel="All for Adoption Speed 0")
        sns.distplot(arr[arr["AdoptionSpeed"] == 1][col], ax=axes[0, 2], axlabel="All for Adoption Speed 1")
        sns.distplot(arr[arr["AdoptionSpeed"] == 2][col], ax=axes[0, 3], axlabel="All for Adoption Speed 2")
        sns.distplot(arr[arr["AdoptionSpeed"] == 3][col], ax=axes[0, 4], axlabel="All for Adoption Speed 3")
        sns.distplot(arr[arr["AdoptionSpeed"] == 4][col], ax=axes[0, 5], axlabel="All for Adoption Speed 4")
        sns.distplot(arr[arr["Type"] == 1][col], ax=axes[1, 0], axlabel="Dog")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 0) & (arr["Type"] == 1)][col], ax=axes[1, 1],
                     axlabel="Dog for Adoption Speed 0")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 1) & (arr["Type"] == 1)][col], ax=axes[1, 2],
                     axlabel="Dog for Adoption Speed 1")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 2) & (arr["Type"] == 1)][col], ax=axes[1, 3],
                     axlabel="Dog for Adoption Speed 2")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 3) & (arr["Type"] == 1)][col], ax=axes[1, 4],
                     axlabel="Dog for Adoption Speed 3")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 4) & (arr["Type"] == 1)][col], ax=axes[1, 5],
                     axlabel="Dog for Adoption Speed 4")
        sns.distplot(arr[arr["Type"] == 2][col], ax=axes[2, 0], axlabel="Cat")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 0) & (arr["Type"] == 2)][col], ax=axes[2, 1],
                     axlabel="Cat for Adoption Speed 0")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 1) & (arr["Type"] == 2)][col], ax=axes[2, 2],
                     axlabel="Cat for Adoption Speed 1")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 2) & (arr["Type"] == 2)][col], ax=axes[2, 3],
                     axlabel="Cat for Adoption Speed 2")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 3) & (arr["Type"] == 2)][col], ax=axes[2, 4],
                     axlabel="Cat for Adoption Speed 3")
        sns.distplot(arr[(arr["AdoptionSpeed"] == 4) & (arr["Type"] == 2)][col], ax=axes[2, 5],
                     axlabel="Cat for Adoption Speed 4")

        for i in range(2, 18):
            plt.close(i)

        plt.show()


def num_boxp(arr, cols):
    for col in cols:
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        f, axes = plt.subplots(1, 3, figsize=[16, 9])
        f.suptitle(col +' by AdopSp.')
        ax1 = sns.catplot(x="AdoptionSpeed", y=col, data=arr, kind="violin", ax=axes[0], col="Type")
        ax2 = sns.catplot(x="AdoptionSpeed", y=col, data=arr[arr["Type"] == 1], kind="violin", ax=axes[1], col="Type")
        ax3 = sns.catplot(x="AdoptionSpeed", y=col, data=arr[arr["Type"] == 2], kind="violin", ax=axes[2], col="Type")
        axes[0].set_title(" For All")
        axes[1].set_title(" For Dogs")
        axes[2].set_title(" For Cats")
        for i in range(2,5):
            plt.close(i)
        plt.show()

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #train, test = read_data()
    #train = conv_cat_variable(train)
    #test = conv_cat_variable(test)
    # sns.catplot(y="Age", x="AdoptionSpeed", data = train, kind = "box")
    #sns.distplot(train.Color3)
    #f, axes = plt.subplots(5, 8, figsize=[16, 9])

    train, test = read_data()

    #ax1 = sns.barplot(x="AdoptionSpeed", data=train, hue="Type")

    x_train, y_train, x_test, id_test = prepare_data(train, test)

    num_boxp(pd.concat([x_train, y_train], axis = 1), Columns.ind_cont_columns.value)



    cols = Columns.ind_num_cat_columns.value
    if "RescuerID" in cols:
        cols.remove("RescuerID")
    #check_cat_perc(pd.concat([x_train, y_train], axis=1, sort=False), cols)
    #check_num_dist(pd.concat([x_train, y_train], axis = 1), Columns.ind_cont_columns.value)
    num_boxp(pd.concat([x_train, y_train], axis = 1), Columns.ind_cont_columns.value)

