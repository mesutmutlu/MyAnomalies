import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
import numpy as np
import matplotlib.pyplot as plt



train = pd.read_csv("C:/Users/dtmemutlu/Downloads/train.csv")
test =  pd.read_csv("C:/Users/dtmemutlu/Downloads/test.csv")
x_train = train.drop("AdoptionSpeed", axis=1)
y_train = train[["AdoptionSpeed"]]
x_test = test.drop("PetID", axis=1)
id_test = test[["PetID"]]
from keras import models, layers, regularizers
from keras.utils.np_utils import to_categorical

from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import cohen_kappa_score

x_train_k = x_train.copy()
#y_train_k = y_train.copy()
x_test_k = x_test.copy()
id_test_k = id_test.copy()

import sys


#y_train_k = to_categorical(y_train_k.values)
if 1 == 0:
    y_train_k = np.zeros((len(y_train), 4))
    i = 0
    for x in y_train["AdoptionSpeed"].values:

        if x == 1:
            y_train_k[i, 0:1] = 1
        elif x == 2:
            y_train_k[i, 0:2] = 1
        elif x == 3:
            y_train_k[i, 0:3] = 1
        elif x == 4:
            y_train_k[i, 0:4] = 1

        #print(x, y_train_k[i])
        i += 1
else:
    y_train_k = to_categorical(y_train, 5)
#print(y_train_k)


import sys


train_cat = np.array([])
test_cat = np.array([])
for c in ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "State", "MaturitySize",
                           "FurLength", "Pet_Breed", "Breed_Merge"]:
    val_cats = np.unique(x_train_k[c].values.tolist() + x_test_k[c].values.tolist())

    ohe = OneHotEncoder(categories=[val_cats], sparse=False)

    print(datetime.now(), c)
    print(x_train_k[c].values.shape)
    x_train_k[c] = x_train_k[c].astype('int32')
    t_cat = ohe.fit_transform(x_train_k[c].values.reshape(-1, 1))
    if train_cat.size == 0:
        train_cat = t_cat
    else:
        train_cat = np.concatenate((train_cat, t_cat), axis=1)
    x_train_k.drop(c, inplace=True, axis=1)

    x_test_k[c] = x_test_k[c].astype('int32')
    te_cat = ohe.fit_transform(x_test_k[c].values.reshape(-1, 1))
    if test_cat.size == 0:
       test_cat = te_cat
    else:
       test_cat = np.concatenate((test_cat, te_cat), axis=1)
    x_test_k.drop(c, inplace=True, axis=1)

    print(train_cat.shape)


print(datetime.now(), "x_train and x_test to ndarray")
x_train_k = np.array(x_train_k.values)
x_test_k = np.array(x_test_k.values)

print(datetime.now(), "concatenate x_train/x_test and train_cat/test_cat")
x_train_k = np.concatenate((x_train_k, train_cat), axis=1)
x_test_k = np.concatenate((x_test_k, test_cat), axis=1)

print(datetime.now(), x_train_k.shape, y_train_k.shape)

from sklearn.model_selection import train_test_split

partial_x_train, x_val, partial_y_train, y_val = train_test_split(x_train_k, y_train_k, test_size=0.20, random_state=42)

print(partial_y_train)
model = models.Sequential()
model.add(layers.Dense(int(160), kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(partial_x_train.shape[1],)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(int(110), kernel_regularizer=regularizers.l2(0.001), activation='relu'))
#model.add(layers.Dropout(0.25))
model.add(layers.Dense(int(77), kernel_regularizer=regularizers.l2(0.001), activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(int(55), kernel_regularizer=regularizers.l2(0.001), activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='sigmoid'))

model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])



history=model.fit(partial_x_train, partial_y_train, epochs=10,
                            batch_size=256, validation_data=(x_val, y_val), shuffle=False)




pred_classes = model.predict_classes(partial_x_train)
print("-------class", pred_classes.shape)
print(pred_classes)

predict = model.predict(partial_x_train)
print("-------predict", predict.shape)
print(predict)

predict_val = model.predict(x_val)


pred_probas = model.predict_proba(partial_x_train)
print("-------probas", pred_probas.shape)
print(pred_probas)

print(partial_y_train)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)
plt.rcParams["figure.figsize"] = [20, 10]
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()




model = models.Sequential()
model.add(layers.Dense(int(128), activation='relu', input_shape=(predict.shape[1],)))
model.add(layers.Dropout(0.20))
model.add(layers.Dense(int(64),  activation='relu'))
model.add(layers.Dense(int(32),  activation='relu'))
#model.add(layers.Dropout(0.25))
model.add(layers.Dense(int(5), activation='sigmoid'))

print(model.summary())

model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


if 1 == 0:
    pd_par_y_train = pd.DataFrame(data=partial_y_train, columns = ["x_"+str(i) for i in range(4)])
    for index, row in pd_par_y_train.iterrows():
        pd_par_y_train["x"] = pd_par_y_train["x_0"] + pd_par_y_train["x_1"] + pd_par_y_train["x_2"] + pd_par_y_train["x_3"]

    pd_y_val = pd.DataFrame(data=y_val, columns = ["x_"+str(i) for i in range(4)])
    for index, row in pd_y_val.iterrows():
        pd_y_val["x"] = pd_y_val["x_0"] + pd_y_val["x_1"] + pd_y_val["x_2"] + pd_y_val["x_3"]
        print(to_categorical(pd_par_y_train["x"].values).shape)
        print(to_categorical(pd_y_val["x"].values).shape)

print(predict.shape)







history=model.fit(predict, partial_y_train, epochs=30,
                            batch_size=12, validation_data=(predict_val, y_val), shuffle=False)

pre = model.predict_proba(predict_val)
print(y_val)
print("------")
print(pre)

#true_labels = np.argmax(y_val, axis=1)

#print(cohen_kappa_score(true_labels,true_labels ))

sys.exit()

pred = model.predict_classes(x_test_k)
id_test_k = np.array(id_test_k.values)
print(id_test_k.shape, pred.shape)
submission = np.concatenate((id_test_k, pred.reshape(-1,1)), axis=1)
print(submission.shape)
print(submission)
submission_df = pd.DataFrame(columns=["PetID, AdoptionSpeed"], data=submission)
print(submission_df.groupby(['AdoptionSpeed']).size())
#submission_df.to_csv('submission.csv', index=False)
