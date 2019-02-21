from scipy.optimize import minimize, rosen, rosen_der

from sklearn.feature_extraction import stop_words

import sys
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from collections import Counter
import numpy as np


if __name__ == "__main__":
    emp = np.empty([3, 3])
    print(emp)
    print(emp[:,1:2])

    print(np.random.randn(5))
    df = pd.DataFrame(columns=["in"], data=[0,1,2,3,4,5])
    import math
    print(round(math.ceil(df["in"].median())))

    sys.exit()


    print(1%3)
    print(type(np.random.random_integers(25,100)))

    layer_sizes = []
    sizes = ()
    for i in range(0,100):
        t = (np.random.random_integers(25,100), np.random.random_integers(25,100), np.random.random_integers(25,100))
        layer_sizes.append(t)

    print(layer_sizes)

    x = np.ones([3,2])
    pd1 = pd.DataFrame(data=x, columns=["x1", "x2"])
    pd1.iloc[1,:] = [2,2]
    pd1x = pd1[pd1["x1"] == 2]
    print(pd1x)

    y = np.zeros([3, 3])
    pd2 = pd.DataFrame(data=y, columns=["y1", "y2", "y3"])
    pd2.iloc[2, :] = [3, 3, 3]
    pd2y = pd2[pd2["y2"] == 3]
    print(pd2y)
    print(pd.concat([pd1x.reset_index(), pd2y.reset_index()], axis=1).drop(["index"], axis=1))