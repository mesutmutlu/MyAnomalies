from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models, layers


if __name__ == "__main__":

    (train_images, train_labes), (test_images, test_labels) = mnist.load_data()

    print(train_images.shape)
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32')/255

    print(test_images.shape)
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32')/255

    train_labes = to_categorical(train_labes)
    test_labels = to_categorical(test_labels)

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    print(model.summary())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    print(model.summary())

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labes, epochs=5, batch_size=64)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)