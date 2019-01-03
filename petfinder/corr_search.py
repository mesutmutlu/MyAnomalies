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
from petfinder.explore import read_data

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
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def by_theilsu(train,indep_cols, dep_cols):
    theilu = pd.DataFrame(index=dep_cols, columns=train[indep_cols].columns)
    columns = train[indep_cols].columns
    for j in range(0, len(columns)):
        u = theils_u(train[dep_cols[0]].tolist(), train[columns[j]].tolist())
        theilu.loc[:, columns[j]] = u
    theilu.fillna(value=np.nan, inplace=True)
    plt.figure(figsize=(20, 1))
    sns.heatmap(theilu, annot=True, fmt='.2f')
    plt.show()

def by_skchi(indep,dep):

    chi2_p = sk_chi(indep, dep)
    res = pd.DataFrame(columns=["Variable", "Chi2 Stat", "P-value", "P-Dependency"])
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

def by_sschi(indep,dep):

    # chi-squared test with similar proportions
    df = pd.concat([indep, dep],axis=1, sort=False)
    #print(df)
    res = pd.DataFrame(columns=["Variable","Degrees of Freedom", "Probability", "Critical Value", "Stats", "Stats Res", "significance", "P-Value", "P Res"])
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

if __name__=="__main__":

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

    corr_y = train[dep_columns]
    corr_x = train[ind_num_cat_columns]

    res = by_skchi(corr_x, corr_y)
    print(res)

    res = by_sschi(corr_x, corr_y)
    print(res)

    by_theilsu(train, ind_num_cat_columns, dep_columns)