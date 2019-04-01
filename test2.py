import pandas as pd
import numpy as np

mic = pd.read_csv(r"C:\Users\dtmemutlu\PycharmProjects\MyAnomalies\df_mic.csv")
spearman = pd.read_csv(r"C:\Users\dtmemutlu\PycharmProjects\MyAnomalies\spearman_for_continous.csv")
chi = pd.read_csv(r"C:\Users\dtmemutlu\PycharmProjects\MyAnomalies\chitest_for_categorical.csv")

s_lst = []
for index, row in spearman.iterrows():
    if (float(row["All"])<=0.15) & (float(row["All"])>=-0.25):
        s_lst.append(row["Feature"])

print("spearman", len(s_lst), len(spearman))
print(s_lst)
#print(mic)
m_lst = []
for index, row in mic.iterrows():
    if (float(row["importance"])<=0.04):
        m_lst.append(row["feature"])

print("mic", len(m_lst), len(mic))
print(m_lst)

i_lst = [value for value in m_lst if value in s_lst]
print("i_lst", len(i_lst))
print(i_lst)
#print(chi)
np_lst = np.array(i_lst)
import math
print(np_lst[:490].reshape((math.ceil(float(len(np_lst[:490]))/float(10))), 10))

lst = []
for index, row in chi.iterrows():
    if (float(row["AdoptionSpeed"])> 0.02):
        lst.append(row["Feature"])

print("chi", len(lst), len(chi))
print(lst)
