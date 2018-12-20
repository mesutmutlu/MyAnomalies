import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys

gender_sub = pd.read_csv("C:/Users/dtmemutlu/PycharmProjects/MyAnomalies/titanic/gender_submission.csv")
train = pd.read_csv("C:/Users/dtmemutlu/PycharmProjects/MyAnomalies/titanic/train.csv")
test =  pd.read_csv("C:/Users/dtmemutlu/PycharmProjects/MyAnomalies/titanic/test.csv")

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
dt.groupby(['Age'])['Survived'].size().plot(kind='bar')
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



