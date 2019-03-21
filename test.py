import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
import numpy as np

arr =np.zeros((2,3))

print( ["pca_"+ str(i) for i in range(arr.shape[1])])



train = pd.read_csv("C:/Users/dtmemutlu/Downloads/train.csv")

from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr, SelectFwe

from sklearn.decomposition import KernelPCA
kPCA = KernelPCA(n_components=150, kernel='rbf')
kPCA.fit(train.drop("AdoptionSpeed", axis=1))
x_train = kPCA.transform(train.drop("AdoptionSpeed", axis=1))
print(x_train.shape)

mic = mutual_info_classif(train.drop(["AdoptionSpeed"], axis=1),train["AdoptionSpeed"], random_state=42)

df_mic = pd.DataFrame(data=mic, index=train.drop(["AdoptionSpeed"], axis=1).columns.values.tolist(),columns=["importance"])

print(df_mic.describe(include="all"))
print(df_mic.sort_values(by=["importance"]))
import sys
sys.exit()
sfpr = SelectFpr(score_func=mutual_info_classif, alpha=0.03)
sfpr.fit(train.drop(["AdoptionSpeed"], axis=1).astype("float"),train["AdoptionSpeed"].astype("float"))
print(sfpr)
print(sfpr.get_support(indices=False))
print(sfpr.get_support(indices=True))
x_fpr = sfpr.transform(train.drop(["AdoptionSpeed"], axis=1))
print(x_fpr)
df_x_fpr = pd.DataFrame(data=x_fpr, index=train.columns[sfpr.get_support(indices=True)], columns=["importance"])
print(df_x_fpr)
print(df_x_fpr.sort_values(by=["importance"]))
#df_x_fpr.sort_values(by=["importance"]).to_csv("x_fpr.csv")
