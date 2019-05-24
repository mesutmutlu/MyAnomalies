import numpy as np

def check_level(x, level=3.5):
    if x >= level:
        return 1
    else:
        return 0

def r_precision(y_pred, y_true, k=None):
    assert (y_pred.ndim == 1)
    assert (y_true.ndim == 1)
    assert (len(y_pred) == len(y_true))
    if (k == None) or (k > len(y_pred)):
        k = len(y_pred)
    y_pred = np.apply_along_axis(check_level, 1, arr=y_pred[:k+1].reshape(-1, 1))
    print(y_pred)
    y_true = np.apply_along_axis(check_level, 1, arr=y_true[:k+1].reshape(-1, 1))
    print(y_true)
    #print(np.dot(y_pred, y_true.T)/len(y_pred))
    return np.dot(y_pred, y_true.T)/len(y_pred)

def r_recall(y_pred, y_true, k=None):
    assert (y_pred.ndim == 1)
    assert (y_true.ndim == 1)
    assert (len(y_pred) == len(y_true))
    if (k == None) or (k > len(y_pred)):
        k = len(y_pred)
    y_pred = np.apply_along_axis(check_level, 1, arr=y_pred[:k+1].reshape(-1, 1))
    y_true = np.apply_along_axis(check_level, 1, arr=y_true.reshape(-1, 1))
    return np.dot(y_pred, y_true.T) / np.sum(y_true)

def r_average_precision(y_pred, y_true, k=None):
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
        print("for loop", i, r_precision(y_pred[:k+1], y_true[:k+1], k=i))
        sum_r_precision += r_precision(y_pred[:k+1], y_true[:k+1], k=i)

    return sum_r_precision/len(y_true)

if __name__ == "__main__":

    tr = np.array([2,3,3.5,2,4,4])
    pr = np.array([1,5,4,3,4,2])

    #print(r_precision(tr,pr))
    #print(r_recall(tr,pr))
    k = 5
    for i in range(k):
        print(r_precision(tr, pr, i))
    print(r_average_precision(tr, pr, k=5))