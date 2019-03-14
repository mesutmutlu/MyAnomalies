from keras.datasets import boston_housing
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt

def normalize_data(train, test):
    mean = train.mean(axis=0)
    train -= mean
    std = train.std(axis=0)
    train /= std

    test -= mean
    test /= std
    return train, test

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    print(train_data.shape, train_targets.shape)
    print(test_data.shape, test_targets.shape)


    train_data, test_data = normalize_data(train_data, test_data)

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []
    if 1 == 0:
        for i in range(k):
            print('processing fold #', i)
            val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
            val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]

            partial_training_data = np.concatenate(
                [train_data[:i * num_val_samples],
                 train_data[(i+1) * num_val_samples:]], axis=0)
            partial_training_targets = np.concatenate(
                [train_targets[:i * num_val_samples],
                 train_targets[(i + 1) * num_val_samples:]], axis=0)

            model = build_model()
            model.fit(partial_training_data, partial_training_targets, epochs=num_epochs, batch_size=1, verbose=0)
            val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
            all_scores.append(val_mae)

            print(all_scores)
            print(np.mean(all_scores))

    num_epochs = 500
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_training_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_training_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]], axis=0)

        model = build_model()
        history = model.fit(partial_training_data, partial_training_targets, epochs=num_epochs, batch_size=1, verbose=0)
        mae_history = history.history['mean_absolute_error']
        all_mae_histories.append(mae_history)
        average_mae_history= [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()
