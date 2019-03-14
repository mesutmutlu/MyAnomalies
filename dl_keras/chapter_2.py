from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def l_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    return (train_images, train_labels), (test_images, test_labels)

def classify_mnist():
    (train_images, train_labels), (test_images, test_labels) = l_mnist()

    print(len(train_images), train_images.shape)
    print(train_labels)

    print(len(test_images), test_images.shape)
    print(test_labels)

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test loss:', test_loss, 'test acc:', test_acc)

def array_shape():

    print("-----0D tensors-----")
    x = np.array(12)
    print(x)
    print(x.ndim)

    print("-----1D tensors-----")
    x = np.array([12, 3, 6, 14])
    print(x)
    print(x.ndim)

    print("-----2D tensors-----")
    x = np.array([[12, 3, 6, 14],
                  [5, 88, 20, 35],
                  [18, 30, 72, 3],])
    print(x)
    print(x.ndim)

    print("-----3D tensors-----")
    x = np.array([[[12, 3, 6, 14],
                  [5, 88, 20, 35],
                  [18, 30, 72, 3]],
                  [[12, 3, 6, 14],
                   [5, 88, 20, 35],
                   [18, 30, 72, 3]],
                  [[12, 3, 6, 14],
                   [5, 88, 20, 35],
                   [18, 30, 72, 3]]])
    print(x)
    print(x.ndim)

    (train_images, train_labels), (test_images, test_labels) = l_mnist()
    print("-----train images-----")
    print(train_images.shape)
    print(train_images.ndim)
    print(train_images.dtype)

    digit = train_images[4]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()


if __name__ == "__main__":
    array_shape()
    pass