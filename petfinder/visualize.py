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


def vis_cat_percentages(arr, indep_cols, dep_col):

    for col in indep_cols:
        #fig, axes = plt.subplots(1, 2)
        df1 = arr[col].value_counts(normalize=True).rename("percentage").mul(100).reset_index()  # .sort_values(col)
        df1.rename(columns={"index":col}, inplace=True)
        ax1 = sns.catplot(x=col, y="percentage", data=df1, kind="bar")
        ax1.set_xticklabels(rotation=90)
        df2 = arr.groupby([col])[dep_col[0]].value_counts(normalize=True).rename('percentage').mul(
            100).reset_index()  # .sort_values(col)
        ax2 = sns.catplot(x=col,y="percentage", data=df2, hue=dep_col[0], kind="bar")
        ax2.set_xticklabels(rotation=90)
        plt.show()

def vis_num_relation(arr, indep_cols, dep_col):

    for col in indep_cols:
        #fig, axes = plt.subplots(1, 2)
        ax1 = sns.boxplot(x=col, y=dep_col[0], data = arr)
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
    vis_num_relation(train, Columns.ind_cont_columns.value, Columns.dep_columns.value)
    #vis_cat_percentages(train, Columns.ind_num_cat_columns.value, Columns.dep_columns.value)
    conv_cat_variable(train)
    #print(train)
    #train.hist()
    #plt.show()