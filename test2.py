import pandas as pd

fm = pd.read_csv(r"C:\Users\dtmemutlu\Downloads\feature_importances (5).csv")

lst = []
for index, row in fm[["feature", "importance"]].iterrows():
    if row["importance"]<=100:
        lst.append(row["feature"])

print(len(lst), len(fm))
print(lst)
