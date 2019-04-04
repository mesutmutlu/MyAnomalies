import pandas as pd
from keras import  layers, models, regularizers
from sklearn.model_selection import RepeatedKFold
import numpy as np
from datetime import datetime
import time
from matplotlib import pyplot as plt
import random
from earthquake.read_file import  get_features

def fit_eval(X_train, y_train, epoch, batch_size, optimizer, activation, dropout):
    kfold = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2652124)
    cv_train_scores = []
    cv_train_loss = []
    cv_val_scores = []
    cv_val_loss = []
    eval = 0

    for train, val in kfold.split(X_train, y_train):
        # create model
        print(datetime.now(), len(train), len(val))
        model = models.Sequential()
        model.add(layers.Dense(150, input_dim=(X_train.shape[1]), activation='relu',
                               kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(150, activation='relu',
                               kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout))
        if activation == "none":
            model.add(layers.Dense(1, activation="relu"))
        else:
            model.add(layers.Dense(1, activation=activation))
        # Compile model
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        # Fit the model
        history = model.fit(X_train.iloc[train].values, y_train.iloc[train].values.ravel(), epochs=epoch, verbose=0,
                            batch_size=batch_size,
                            validation_data=(X_train.iloc[val].values, y_train.iloc[val].values.ravel()))
        history_dict = history.history
        # collect score and loss values
        # print(history_dict)
        cv_train_loss.append(history_dict['loss'])
        cv_val_loss.append(history_dict['val_loss'])
        cv_train_scores.append(history_dict['mean_absolute_error'])
        cv_val_scores.append(history_dict['val_mean_absolute_error'])
        eval += model.evaluate(X_train.iloc[val].values, y_train.iloc[val].values.ravel(), verbose=0)[1]

    if 1 == 0:
        pred = model.predict(X_train)
        print(pred.shape)
        pred = np.abs(pred)
        print(y_train["time_to_failure"].values.reshape(-1, 1))
        pred_df = pd.DataFrame(data=np.concatenate((y_train["time_to_failure"].values.reshape(-1, 1), pred), axis=1),
                               columns=["gt", "pr"])

    if 1 == 0:
        print(pred_df.to_csv("./output/pred.csv"))
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

    return epoch, batch_size, optimizer, activation, float(eval/10.0)

if __name__ == "__main__":

    X_train, y_train, X_test = get_features()

    pd.concat([X_train, y_train], axis=1).corr()[["time_to_failure"]].to_csv("./output/pearson_corr.csv")


    epochs = [250, 500, 1000, 1500, 2000, 2500]
    batch_sizes = [64, 128, 256]
    optimizers = ["adam", "rmsprop"]
    activations= ["relu", "none"]
    dropouts = [0.1, 0.2, 0.3, 0.4]
    n_grid = 10
    scores = pd.DataFrame(index=[i for i in range(n_grid)],
                          columns=["epoch", "batch_size", "optimizer", "activation", "dropout", "score", "start_date", "end_date"])
    for i in range(n_grid):
        r_epoch = random.choice(epochs)
        r_batch_size = random.choice(batch_sizes)
        r_optimizer = random.choice(optimizers)
        r_activation = random.choice(activations)
        r_dropout = random.choice(dropouts)
        start_date = datetime.now()
        print(i, datetime.now(), r_epoch, r_batch_size, r_optimizer, r_activation, r_dropout)
        epoch, batch_size, optimizer, activation, score = \
            fit_eval(X_train, y_train, r_epoch, r_batch_size, r_optimizer, r_activation, r_dropout)
        end_date = datetime.now()

        scores.loc[i, "epoch"] = epoch
        scores.loc[i, "batch_size"] = batch_size
        scores.loc[i, "optimizer"] = optimizer
        scores.loc[i, "activation"] = activation
        scores.loc[i, "score"] = score
        scores.loc[i, "start_date"] = start_date
        scores.loc[i, "end_date"] = end_date

    scores.sort_values(by=["score"]).to_csv("./output/2hdscores_150-150units.csv", index=False)


