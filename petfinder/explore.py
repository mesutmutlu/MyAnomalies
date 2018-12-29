import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt




def read_data():

    train = pd.read_csv("C:/datasets/petfinder.my/train/train.csv")
    return train


if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    train = read_data()
    #print(train.head())
    print(train.describe(include="all"))
    #print(train.info())
   # print(train[train.Name.isnull()])
    #sns.countplot(x=train.Vaccinated , hue=train.AdoptionSpeed)
    sns.catplot(x="AdoptionSpeed", col="Type", data=train, kind="count", hue="Gender")
    #sns.pairplot(train[["AdoptionSpeed", "Gender", "Type"]], hue="Type")
    #g.despine(left=True)
    #g.set_ylabels("adoption per sex")
    plt.show()