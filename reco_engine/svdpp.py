import numpy as np
import pandas as pd
from functools import partial
import scipy as sp
from scipy.optimize import minimize
import sys

def calc_mean(np_arr):
    results = []
    #print(np_arr[..., 0])
    for x in sorted(np.unique(np_arr[..., 0])):
        #print(np_arr[np.where(np_arr[...,0] == x)][...,1])
        results.append([x, np.average(np_arr[np.where(np_arr[...,0] == x)][...,1])])
    return np.array(results)

def calc_std(np_arr):
    results = []
    for x in sorted(np.unique(np_arr[..., 0])):
        #print(np_arr[np.where(np_arr[...,0] == x)][...,1])
        results.append([x, np.std(np_arr[np.where(np_arr[...,0] == x)][...,1])])
    return np.array(results)

metric1 = 0

def to_bu_bi(vec, ln):
    tl_bu_bi = (2 * ln) + 1
    l_bu_bi = int((tl_bu_bi - 1) / 2)

    l = vec[tl_bu_bi - 1]
    x_bu = vec[0:l_bu_bi].reshape(-1, 1)
    x_bi = vec[l_bu_bi:tl_bu_bi - 1].reshape(-1, 1)
    return x_bu, x_bi, l

def loss(bu_bi, X):
    x_bu, x_bi, l = to_bu_bi(bu_bi, len(X))

    mean = np.mean(X[:, 2])
    explicit_factor = np.sum([X[:,2].reshape(-1,1), -1*mean, -1*x_bu, -1*x_bi])
    print(explicit_factor)
    return (np.dot(np.transpose(explicit_factor), explicit_factor) +
            l*(np.dot(np.transpose(x_bu),x_bu) + np.dot(np.transpose(x_bi),x_bi)))[0,0]




def loss2(x, ratings, ln):

    x1 = x[0:ln].reshape(-1,1)
    x2 = x[ln:ln*2].reshape(-1,1)
    mean = np.mean(y)
    l = x[ln*2]
    explicit_factor = np.sum([np.array(y).reshape(-1, 1), -1 * mean, -1 * x1, -1 * x2])

    return np.dot(np.transpose(explicit_factor), explicit_factor) + l*(np.dot(np.transpose(x1),x1) + np.dot(np.transpose(x2),x2))


if __name__ == "__main__":



    ratings = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_ratings.csv")[["userId","id","rating"]].values

    bu = calc_std(ratings[:, [0,2]])
    bi = calc_std(ratings[:, [1,2]])
    mean = np.mean(ratings[:,2])
    #print(bu)
    #print(bi)
    l = 100
    #print(mean)
    #print(y)
    #print(loss(bu, bi, l, mean, ratings))
    # equal initial weighting
    #initial_guess = np.concatenate((bu, bi), axis=1)
    #print(bu)
   # res = sp.optimize.minimize(loss2, [1,2])
   # print(res.x)
    #sys.exit()
    x_bu = ratings[:, [0, 2]]
    x_bi = ratings[:, [1, 2]]
    for x in bu:
        x_bu[x_bu[:, 0] == x[0], 1] = x[1]
    x_bu = x_bu[:, 1].reshape(-1, 1)

    #print(x_bu.shape)
    for x in bi:
        x_bi[x_bi[:, 0] == x[0], 1] = x[1]
    x_bi = x_bi[:, 1].reshape(-1, 1)
    bu_bi = np.concatenate((x_bu, x_bi), axis=0)
    #print(x_bi.shape)
    #print(ratings.shape)

    bu_bi = np.append(bu_bi.flatten(), [100]).ravel().tolist()

    x = bu_bi
    print(len(x), type(x))
    print(x)
    y = ratings[:,2].ravel().tolist()
    print(len(y), type(y))
    print(y)
    loss2_partial = partial(loss2, ratings=ratings, ln=len(y))

    res = sp.optimize.minimize(loss2_partial, x)
    print(res.x)
    sys.exit()

    print(bu_bi)
    #sys.exit()
    loss_partial = partial(loss, X=ratings)
    res = sp.optimize.minimize(loss_partial, bu_bi, options={'disp':True})
    #print(res.x)

