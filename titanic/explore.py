from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys
import numpy as np
import re


def read_data():
    gender_sub = pd.read_csv("C:/Users/dtmemutlu/PycharmProjects/MyAnomalies/titanic/gender_submission.csv")
    train = pd.read_csv("C:/Users/dtmemutlu/PycharmProjects/MyAnomalies/titanic/train.csv")
    test = pd.read_csv("C:/Users/dtmemutlu/PycharmProjects/MyAnomalies/titanic/test.csv")
    return train, test, gender_sub

def analyze_age():
    train, test, gender_sub = read_data()
    dt = train.groupby(['Age'])['Survived'].agg(['count', 'sum'])
    dt["perc"] = dt["sum"] / dt["count"]
    train["Age"].plot(kind="box")
    # dt["perc"].plot(kind="bar", secondary_y="perc")
    train.groupby(['Age'])['Survived'].agg(['mean']).plot(kind="bar")

    train.groupby(['Age'])['Survived'].agg(['mean']).plot(kind="density")
    # dt["perc"].replace("inf", "0")
    print(dt)
    plt.show()

def fill_age_na():

    train, test, gender_sub = read_data()
    print(train["Age"].describe())
    mean = round(train["Age"].mean(),2)
    std = round(train["Age"].std(), 2)
    is_null = train["Age"].isnull().sum()
    rand_age = np.random.randint(mean - 2*std, mean + 2*std, size=is_null)
    age_slice = train["Age"].copy()
    age_slice[np.isnan(age_slice)]=rand_age
    train["Age"] = age_slice

    #print(mean, std, is_null, rand_age)
    print(train["Age"].describe())
    return train

def analyze_cabin():
    train = fill_cabin_na()
    print(train)
    train.groupby(['Deck'])['Survived'].agg(['count', 'sum', 'mean']).plot(kind="bar", secondary_y="mean")
    #train["Deck"].plot(kind="box")
    # dt["perc"].plot(kind="bar", secondary_y="perc")
    train.groupby(['Deck'])['Survived'].agg(['mean']).plot(kind="bar")

    train.groupby(['Deck'])['Survived'].agg(['mean']).plot(kind="density")
    # dt["perc"].replace("inf", "0")
    #print(train)
    plt.show()

def fill_cabin_na():
    deck = {"T":0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    train, test, gender_sub = read_data()
    train['Cabin'] = train['Cabin'].fillna("U0")
    train['Deck'] = train['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group(0))
    train['Deck'] = train['Deck'].map(deck)
    print(train['Deck'].describe())

    return train
    #print(pd.concat([g, h]).drop_duplicates(keep=False))


def test():
    train, test, gender_sub = read_data()

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(train.describe(include='all'))

    dt = train.drop(['Name','Ticket', 'PassengerId'], axis=1)
    le = LabelEncoder()
    #le.fit(train['Sex'].unique())
    #print(le.classes_)
    #print(dt['Sex'].shape)
    dt['SexN'] = le.fit_transform(dt['Sex'])
    #print(dt.Cabin.unique())
    categorical_feature_mask = dt.dtypes==object
    categorical_cols = dt.columns[categorical_feature_mask].tolist()
    #print(dt.groupby(['Sex', 'Embarked'])['Survived'].mean())
    #fig = plt.figure()

    print(dt.groupby(['Age'])['Survived'].agg(['count', 'sum']))
    print(dt['Survived'])
    dt.groupby(['SibSp'])['Survived'].agg(['count', 'sum']).plot(kind='bar')
    #dt["Fare"][dt["Survived"]==1].plot(kind="bar")
    plt.show()

    sys.exit()

    fig, axes = plt.subplots(nrows=4, ncols=4)


    p1= dt.groupby(['Sex', 'Embarked'])['Survived'].size().unstack().plot(kind='bar',  ax=axes[0,0])
    p2= dt.groupby(['Sex', 'Embarked'])['Survived'].mean().unstack().plot(kind='bar', ax=axes[1,0])
    p3= dt.groupby(['Embarked', 'Sex'])['Survived'].size().unstack().plot(kind='bar',  ax=axes[0,1])
    p4 = dt.groupby(['Embarked', 'Sex'])['Survived'].mean().unstack().plot(kind='bar', ax=axes[1,1])
    p5 = dt.groupby(['Sex','Pclass'])['Survived'].size().unstack().plot(kind='bar',  ax=axes[0,2])
    p6 = dt.groupby(['Sex','Pclass'])['Survived'].mean().unstack().plot(kind='bar', ax=axes[1,2])

    bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]
    dt['Age'].fillna(dt['Age'].mean(),inplace=True)
    p5 = dt.groupby(['Age']['Survived'])['Survived'].size().plot(kind='bar',  ax=axes[0,3], xticks = bins)
    p6 = dt.groupby(['Age']['Survived'])['Survived'].mean().plot(kind='bar', ax=axes[1,3], xticks = bins)
    p7 = dt['Age'][dt["Survived"]==1].hist( ax=axes[2,0])
    p8 = dt['Age'][dt["Survived"]==0].hist( ax=axes[3,0])
    p9 = dt['Age'][dt["Sex"]=="female"].hist( ax=axes[2,1])
    p10 = dt['Age'][dt["Sex"]=="male"].hist( ax=axes[3,1])
    #print(dt['Age'][dt["Sex"]=="female"])
    #p11 = dt['Age'].groupby(['Survived'])['Age'].size().plot(kind='bar',  ax=axes[0,3], xticks = bins)
    #p12 = dt['Age'].groupby(['Survived'])['Age'].mean().plot(kind='bar', ax=axes[1,3], xticks = bins)

    #p9 = dt['Age'][dt["Survived"]==1 & dt["Sex"]=="female"].hist( ax=axes[2,2])
    #p10 = dt['Age'][dt["Survived"]==0 & dt["Sex"]=="female"].hist( ax=axes[3,2])

    print(p1.patches)
    for p in p1.patches:
        print(p)
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        print(x,y)
        #x.annotate('{:.0f} %'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))
    plt.show()
    #dt[categorical_cols] = dt[categorical_cols].apply(lambda col: le.fit_transform(col))

    #arr=["paris", "paris", "tokyo", "amsterdam"]
    #print(arr.shape)

if __name__ == "__main__":
    analyze_cabin()
