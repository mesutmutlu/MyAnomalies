import numpy as np
from functools import partial
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

def check_level(x, level=3.5):
    if x >= level:
        return 1
    else:
        return 0


def r_precision(y_true, y_pred, k=None, level=3.5):
    assert (y_pred.ndim == 1)
    assert (y_true.ndim == 1)
    assert (len(y_pred) == len(y_true))
    if (k is None) or (k > len(y_pred)):
        k = len(y_pred)
    partial_check_level = partial(check_level, level=level)
    y_pred = np.apply_along_axis(partial_check_level, 1, arr=y_pred[:k+1].reshape(-1, 1))
    #print(y_pred)
    y_true = np.apply_along_axis(partial_check_level, 1, arr=y_true[:k+1].reshape(-1, 1))
    #print(y_true)
    #print(np.dot(y_pred, y_true.T)/len(y_pred))
    return np.dot(y_pred, y_true.T)/len(y_pred)


def r_recall(y_true, y_pred, k=None, level=3.5):
    assert (y_pred.ndim == 1)
    assert (y_true.ndim == 1)
    assert (len(y_pred) == len(y_true))
    if (k is None) or (k > len(y_pred)):
        k = len(y_pred)
    partial_check_level = partial(check_level, level=level)
    y_pred = np.apply_along_axis(partial_check_level, 1, arr=y_pred[:k+1].reshape(-1, 1))
    print(y_pred)
    y_true = np.apply_along_axis(partial_check_level, 1, arr=y_true[:k+1].reshape(-1, 1))
    print(y_true)
    return np.dot(y_pred, y_true.T) / np.sum(y_true)


def map_k(y_true, y_pred, k=None):
    assert (y_pred.ndim == 1)
    assert (y_true.ndim == 1)
    assert (len(y_pred) == len(y_true))
    if (k == None) or (k > len(y_pred)):
        k = len(y_pred)

    # y_true_ = np.apply_along_axis(check_level, 1, arr=y_true[:k+1].reshape(-1, 1))
    #
    # y_true_pred = y_pred.reshape(-1, 1)*y_true_.reshape(-1,1)
    # print(y_pred.reshape(-1, 1))
    # print(y_true_.reshape(-1,1))
    # print(y_true_pred)

    sum_r_precision = 0
    for i in range(k):
        #print("for loop", i, r_precision(y_pred[:k+1], y_true[:k+1], k=i))
        sum_r_precision += r_precision(y_pred[:k+1], y_true[:k+1], k=i)
        #print(sum_r_precision)

    return sum_r_precision/k


def mar_k(y_true, y_pred, k=None):
    assert (y_pred.ndim == 1)
    assert (y_true.ndim == 1)
    assert (len(y_pred) == len(y_true))
    if (k == None) or (k > len(y_pred)):
        k = len(y_pred)

    # y_true_ = np.apply_along_axis(check_level, 1, arr=y_true[:k+1].reshape(-1, 1))
    #
    # y_true_pred = y_pred.reshape(-1, 1)*y_true_.reshape(-1,1)
    # print(y_pred.reshape(-1, 1))
    # print(y_true_.reshape(-1,1))
    # print(y_true_pred)

        sum_r_recall = 0
    for i in range(k):
        #print("for loop", i, r_precision(y_pred[:k+1], y_true[:k+1], k=i))
        sum_r_recall += r_recall(y_pred[:k+1], y_true[:k+1], k=i)
        #print(sum_r_precision)

    return sum_r_recall/k


#defines if the list has compatible items based on item's attributes
def intra_list_similarity():
    pass

#assess if system recommends many of the same items to different users
#calculated using distance of recommendation lists of different items
def personalization(predictions):
    #predictions as binary values:0,1
    similarity_mean = np.mean(1-np.triu(cosine_similarity(predictions),1))
    return 1 - similarity_mean

#Coverage is the percent of items in the training data the model is able to recommend on a test set
def coverage():
    pass

if __name__ == "__main__":

    tr = np.array([4,4,5,2,4,4])
    pr = np.array([3,4,5,3,4,2])
    print(personalization([[1,0,0],[1,1,0],[0,1,0]]))
    print(r_precision(tr, pr))
    print(r_recall(tr, pr, level=3.9))
    k = 3
    #for i in range(k):
    #    print(r_precision(tr, pr, i))
    print(map_k(tr, pr, k=3))