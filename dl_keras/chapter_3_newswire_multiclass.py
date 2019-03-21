from keras.datasets import reuters
from keras import models, layers
import numpy as np
import  matplotlib.pyplot as plt
import keras.backend as K
from sklearn.metrics import  cohen_kappa_score
import tensorflow as tf
import sys


def kappa_loss(y_true, y_pred):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    print("---------")
    print(y_true)
    sess = tf.Session()
    print('sess.run')
    print(sess.run(y_true))
    with sess.as_default():
        print(y_true.shape)
        y_true = y_true.eval()
        y_pred = y_pred.eval()
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    # Return a function

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=46):
        results = np.zeros((len(labels), dimension))
        for i, label in enumerate(labels):
            results[i, label] = 1.
        return results
if __name__ == '__main__':

    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    print(len(train_data))
    print(len(test_data))

    print(train_data)
    print(train_labels)

    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decoded_newswire)

    print(train_labels[10])

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    one_hot_train_labels = to_one_hot(train_labels)
    one_hot_test_labels = to_one_hot(test_labels)
    print(one_hot_train_labels)

    if 1 == 0:

        from keras.utils.np_utils import to_categorical

        one_hot_train_labels = to_categorical(train_labels)
        one_hot_test_labels = to_categorical(test_labels)
        print(one_hot_train_labels)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]
    if 1 == 1:
        history = model.fit(partial_x_train, partial_y_train, epochs=20,
                            batch_size=512, validation_data=(x_val, y_val))

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(loss)+1)

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

    model.fit(partial_x_train, partial_y_train, epochs=9,
                        batch_size=512, validation_data=(x_val, y_val))
    pred = model.predict_classes(x_val)
    print(pred)
    print("--------")
    print(y_val)
    print(cohen_kappa_score(pred, train_labels[:1000], weights='quadratic'))
    #print(history.history)
    sys.exit()
    results = model.evaluate(x_test, one_hot_test_labels)
    print(results)

    predictions = model.predict(x_test)

    print(predictions[0].shape)
    print(np.sum(predictions[0]))
    print(np.argmax(predictions[0]))