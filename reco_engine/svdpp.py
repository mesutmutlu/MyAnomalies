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


def loss(bu_bi, X):

    tl_bu_bi = len(bu_bi)
    l_bu_bi = int((tl_bu_bi-1)/2)

    l = bu_bi[tl_bu_bi-1]
    x_bu = bu_bi[0:l_bu_bi].reshape(-1,1)
    x_bi = bu_bi[l_bu_bi:tl_bu_bi-1].reshape(-1,1)
    mean = np.mean(ratings[:, 2])

    explicit_factor = np.sum([X[:,2].reshape(-1,1), mean, -1*x_bu, -1*x_bi])
    return (np.dot(np.transpose(explicit_factor), explicit_factor) +
            l*(np.dot(np.transpose(x_bu),x_bu) + np.dot(np.transpose(x_bi),x_bi)))[0,0]


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

    bu_bi = np.append(bu_bi, [100])
    print(bu_bi)
    #sys.exit()
    loss_partial = partial(loss, X=ratings)
    res = sp.optimize.minimize(loss_partial, bu_bi, method='Nelder-Mead', options={'maxiter':5, 'disp':True})
    print(res.x)

