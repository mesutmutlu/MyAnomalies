import pandas as pd
from keras import  layers, models, regularizers
from sklearn.model_selection import RepeatedKFold
import numpy as np
from datetime import datetime
import time
from matplotlib import pyplot as plt

X_train = pd.read_csv(r"C:\datasets\earthquake\xtrain.csv")
X_train.drop("Unnamed: 0", inplace=True, axis=1)
print(X_train.head())
y_train = pd.read_csv(r"C:\datasets\earthquake\ytrain.csv")
y_train.drop("Unnamed: 0", inplace=True, axis=1)
print(y_train.head())

for c in (X_train.columns.values.tolist()):
    mean = X_train[c].mean()
    X_train[c] = X_train[c] - mean
    #X_test[c] = X_test[c] - mean
    std = X_train[c].std()
    X_train[c] = X_train[c] / std
    #X_test[c] = X_test[c] / std

pd.concat([X_train, y_train], axis=1).corr()[["time_to_failure"]].to_csv("./output/pearson_corr.csv")


kfold = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2652124)
cv_train_scores = []
cv_train_loss = []
cv_val_scores = []
cv_val_loss = []

cv_history = pd.DataFrame(dtype=np.float64, index=range(1, 11))

for train, val in kfold.split(X_train, y_train):
    # create model
    print(datetime.now(), len(train), len(val))
    model = models.Sequential()
    model.add(layers.Dense(150, input_dim=(X_train.shape[1]), activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(150, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))
    # Compile model
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    # Fit the model
    history = model.fit(X_train.iloc[train].values, y_train.iloc[train].values.ravel(), epochs=200, verbose=0,
                        batch_size=128, validation_data=(X_train.iloc[val].values, y_train.iloc[val].values.ravel()))
    history_dict = history.history
    # collect score and loss values
    # print(history_dict)
    cv_train_loss.append(history_dict['loss'])
    cv_val_loss.append(history_dict['val_loss'])
    cv_train_scores.append(history_dict['mean_absolute_error'])
    cv_val_scores.append(history_dict['val_mean_absolute_error'])

pred = model.predict(X_train)
print(pred.shape)
print(y_train["time_to_failure"].values.shape)
print(np.concatenate((y_train["time_to_failure"].values,pred), axis=1))
print(np.min(pred))
# sum score and loss values for different CVs
train_loss = [sum(x) for x in zip(*cv_train_loss)][80:]
val_loss = [sum(x) for x in zip(*cv_val_loss)][80:]
train_scores = [sum(x) for x in zip(*cv_train_scores)][80:]
val_scores = [sum(x) for x in zip(*cv_val_scores)][80:]

plt.figure(figsize=(25, 10))
plt.plot(range(1, len(train_loss) + 1), train_loss, 'bo', label='Training loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, 'b', label='Validation loss')
plt.title("Trainin and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.figure(figsize=(25, 10))
plt.plot(range(1, len(train_scores) + 1), train_scores, 'bo', label='Training mae')
plt.plot(range(1, len(val_scores) + 1), val_scores, 'b', label='Validation mae')
plt.title("Trainin and validation mae")
plt.xlabel('Epochs')
plt.ylabel('Mae')
plt.legend()

plt.show()


