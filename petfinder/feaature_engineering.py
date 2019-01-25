from sklearn import manifold
from petfinder.get_explore import read_data
from petfinder.preprocessing import prepare_data
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime

def tsne():
    pass


if __name__ == "__main__":

    train, test = read_data()
    x_train, y_train, x_test, y_test = prepare_data(train, test)
    perplexities = [5, 30, 50, 100]
    (fig, subplots) = plt.subplots(1, 5, figsize=(15, 8))

#    ax = subplots[0][0]
    print("tsne started")

    for i, perplexity in enumerate(perplexities):
        #ax = subplots[0][i + 1]

        t0 = time()
        print(datetime.datetime.now())
        tsne = manifold.TSNE(n_components=2, init='random',
                             random_state=0, perplexity=perplexity)
        print(datetime.datetime.now())
        Y = tsne.fit_transform(x_train)
        print(datetime.datetime.now())
        t1 = time()
        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        print(Y)
        data = pd.concat([Y, y_train.values], axis=1)
        print(data)
        #ax.set_title("Perplexity=%d" % perplexity)
        sns.scatterplot(x=data[0], y=data[1], hue="AdoptionSpeed",  data=data, ax=ax[0,i])

        #ax.axis('tight')