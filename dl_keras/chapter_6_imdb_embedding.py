import os
imdb_dir = "C:/datasets/aclImdb"
train_dir = os.path.join(imdb_dir, "train")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras import models, layers


if __name__ == "__main__":

    labels = []
    texts = []

    for label_type in  ["neg", "pos"]:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == ".txt":
                f = open(os.path.join(dir_name, fname), encoding="utf8")
                texts.append(f.read())
                f.close()
                if label_type == "neg":
                    labels.append(0)
                else:
                    labels.append(1)

    maxlen = 100
    training_samples = 200
    validation_samples = 10000
    max_words = 10000

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    print("sequences fff")
    print(sequences)
    word_index = tokenizer.word_index

    print("Found %s unique tokens." % len(word_index))
    print("word_index")
    print(word_index)

    data = pad_sequences(sequences, maxlen = maxlen)

    labels = np.asarray(labels)
    print("Shape of data tensor:", data.shape)
    print("Shape of label tensor:", labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples+validation_samples]
    y_val = labels[training_samples: training_samples+validation_samples]

    glove_dir = "C:/datasets/aclImdb/glove.6B"

    embeddings_index = {}
    f = open(os.path.join(glove_dir, "glove.6B.100d.txt"), encoding="utf8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    print("Found %s word vectors." % len(embeddings_index))
    #print(embeddings_index)

    embedding_dim = 100

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None :
                embedding_matrix[i] = embedding_vector

    model = models.Sequential()
    model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    print(model.summary())

    #load the pretrained embedding weights, can be also done without these 2 lines
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs = 10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    model.save_weights('./output/pre_trained_glove_model.h5')