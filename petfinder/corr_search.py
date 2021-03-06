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
from petfinder.preprocessing import prepare_data
from scipy.stats import ttest_ind, f_oneway, normaltest, ks_2samp, kruskal
import datetime
from petfinder.get_explore import read_data
from petfinder.get_explore import Paths
from petfinder.get_explore import Columns


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def conditional_entropy(x,y):
    # for categorical vs numerical correlation
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy


def theils_u(x, y):
    # for categorical correlation
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def by_theilsu(train,cols):
    # for categorical correlation
    theilu = pd.DataFrame(index=cols, columns=cols)
    i = 0
    for c1 in cols:
        for c2 in cols:
            u = theils_u(train[c1].tolist(), train[c2].tolist())
            theilu.loc[c1, c2] = u
        i = i+1
    theilu.fillna(value=np.nan, inplace=True)
    #plt.figure(figsize=(20, 1))
    print(theilu)
    #result = theilu.pivot(index='SymmetricDivision', columns='MutProb', values='value')
    sns.heatmap(theilu, annot=True, fmt='.2f')
    plt.show()


def correlation_ratio(categories, measurements):
    # for mix correlation
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta


def by_correlation_ratio(df, dep_col, cols):
    # for mix correlation
    #print(df)
    cr = pd.DataFrame(columns=cols)
    i = 0
    for c in cols:
        eta = correlation_ratio(df[dep_col], df[c])
        cr.loc[i, c] = eta
        #i = i + 1
    cr.fillna(value=np.nan, inplace=True)
    # plt.figure(figsize=(20, 1))
    return cr



def by_skchi(indep, dep):

    chi2_p = sk_chi(indep, dep)
    res = pd.DataFrame(columns=["Variable", "Chi2 Stat", "P-value", "Dependency"])
    i=0
    for col in list(indep.columns.values):
        #print(col)
        res.loc[i] = [col,chi2_p[0][i],chi2_p[1][i],chi2_p[1][i] < 1e-2]
        i=i+1
    # print(res)
    return res


def cramers_stat(indep, dep):
    i = 0
    res = pd.DataFrame(columns=["independent", "stat"])
    for col in list(indep.columns.values):
        #print(col)
        confusion_matrix = pd.crosstab(indep[col], dep)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))
        #print(col, np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1))) )
        res.loc[i] = [col,np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1))) ]
        i = i + 1
    return res


def cramers_corrected_stat(indep, dep):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    i = 0
    res = pd.DataFrame(columns=["independent", "stat"])
    for col in list(indep.columns.values):
        confusion_matrix = pd.crosstab(indep[col], dep)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        res.loc[i] = [col, np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))]
        i = i + 1
    return res


def by_sschi(indep,dep):

    # chi-squared test with similar proportions
    df = pd.concat([indep, dep],axis=1, sort=False)
    res = pd.DataFrame(columns=["col", "dof", "prob", "critical", "stat", "s_res", "alpha", "p" , "p_res"])
    print(res)
    i=0
    for col in list(indep.columns.values):
        table = pd.crosstab(indep[col], dep.AdoptionSpeed)
        stat, p, dof, expected = chi2_contingency(table)
        #print('dof=%d' % dof)
        #print(expected)
        # interpret test-statistic
        prob = 0.99
        critical = chi2.ppf(prob, dof)
        #print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
        if abs(stat) >= critical:
            s_res= 'Dependent (reject H0)'
        else:
            s_res = 'Independent (fail to reject H0)'
        # interpret p-value
        alpha = 1.0 - prob
        #print('significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            p_res = 'Dependent (reject H0)'
        else:
            p_res = 'Independent (fail to reject H0)'
        res.loc[i] = [col, dof, prob, critical, stat, s_res, alpha, p , p_res]
        i = i + 1

    return res


def by_paired_ttest(arr):

    for c in arr.columns.values:
        i = 0
        for i in range(0, len(arr.columns.values)):
            #stats = ttest_rel(arr[c], arr[i])
            print(c, arr[i].name)
    pass


def check_samples_diff(df, dep_col):
    res = pd.DataFrame(columns=["Column", "Value1", "Value2", "P-Value", "Difference?"])
    i = 0
    for c1 in df.columns.values:
        print(c1, datetime.datetime.now())
        #stats = ttest_rel(arr[c1], arr[c2])
        values = df[c1].unique()
        import itertools
        combinations = list(itertools.combinations(values, 2))

        for comb in combinations:
            ds1 = df[df[c1] == comb[0]][dep_col]
            ds2 = df[df[c1] == comb[1]][dep_col]
            ds1_p = 0
            ds2_p = 0
            if len(ds1) > 8 & len(ds2) > 8:
                ds1_p = normaltest(ds1)[1]
                ds2_p = normaltest(ds2)[1]

            #print(type(args[0]))
            if (ds1_p > 5e-2) & (ds2_p > 5e-2):
                #come from normal distribution so apply 2 samples student t-test
                print(c1, comb[0], "normal distribution")
                p_value = ttest_ind(ds1, ds2)[0]
            else:
                # come from non normal distribution so apply 2 samples Kolmogorov-Smirnov statistic
                p_value = ks_2samp(ds1, ds2)[0]

            if p_value < 5e-2:
                res.loc[i] = [c1, comb[0], comb[1],p_value , p_value < 5e-2 ]
                i = i + 1
    return res


# check adoption speed distributions over categorical variables
def check_samples_diff2(df, dep_col):
    res = pd.DataFrame(columns=["Column", "Value1", "Value2", "P-Value", "Difference?"])
    i = 0
    for c1 in df.columns.values:
        print(c1, datetime.datetime.now())
        #stats = ttest_rel(arr[c1], arr[c2])
        values = df[c1].unique()
        import itertools
        combinations = list(itertools.combinations(values, 2))

        for comb in combinations:
            ds1 = df[df[c1] == comb[0]][dep_col]
            ds2 = df[df[c1] == comb[1]][dep_col]
            ds1_p = 0
            ds2_p = 0
            if len(ds1) > 8 & len(ds2) > 8:
                ds1_p = normaltest(ds1)[1]
                ds2_p = normaltest(ds2)[1]

            #print(type(args[0]))
            if (ds1_p > 5e-2) & (ds2_p > 5e-2):
                #come from normal distribution so apply 2 samples student t-test
                print(c1, comb[0], "normal distribution")
                p_value = ttest_ind(ds1, ds2)[0]
            else:
                # come from non normal distribution so apply 2 samples Kolmogorov-Smirnov statistic
                p_value = ks_2samp(ds1, ds2)[0]

            if p_value < 5e-2:
                res.loc[i] = [c1, comb[0], comb[1],p_value , p_value < 5e-2 ]
                i = i + 1
    return res

def by_kruskal(arr, dep_col):
#check adoption speed median for different categories, each category value should have at least 5 measurements
    df =  pd.DataFrame(columns=arr.columns.values)

    for c in arr.columns.values:
        if c != dep_col:
            print("-------Kruskal Wallis H-test test for :--------", c)
            arg = []
            for v in arr[c].unique():
                #print("number of measurements of value", v, "on column", c, "is", len(arr[arr[c] == v]))
                if (len(arr[arr[c] == v]) >= 5):
                    #print(arr[arr[dep_col] == v][c].head())
                    arg.append(arr[arr[c] == v][dep_col])
            print("number of categorical groups having more than 5 ameasurements found is ", len(arg), "/",len(arr[c].unique()) )
            if len(arg)>=2:
                H, pval = kruskal(*arg)

                print(c, "H-statistic:", H, "P-Value:", pval)

                if pval < 0.05:
                    print("Reject NULL hypothesis - Significant differences exist between groups.")
                    df.loc[0, c] = 1
                if pval > 0.05:
                    print("Accept NULL hypothesis - No significant difference between groups.")
                    df.loc[0, c] = 0
            else:
                print("Not Tested Because of lack of at least 2 groupes having minimum 5 measurements")
                df.loc[0, c] = -1

    print(df)
    #sns.heatmap(df, annot=True)
    #plt.show()

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train, test = read_data()
    x_train, y_train, x_test, id_test = prepare_data(train, test)
    f_train = pd.concat([x_train, y_train], axis=1)
    #arg = []
    #arg.append(train[train["Type"]==1]["AdoptionSpeed"])
    #arg.append(train[train["Type"]==2]["AdoptionSpeed"])
    print(kruskal(train[train["Type"]==1]["AdoptionSpeed"], train[train["Type"]==2]["AdoptionSpeed"]))
    #print(kruskal(*arg))
    by_kruskal(f_train[Columns.ind_num_cat_columns.value+Columns.dep_columns.value], Columns.dep_columns.value[0])

    for c in Columns.ind_cont_columns.value :
        by_kruskal(f_train[[c] + Columns.dep_columns.value], c)
