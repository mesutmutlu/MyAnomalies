import numpy as np
import scipy.stats as st

def zscore(x):

    if x.ndim == 0:
        return x
    else:
        print(x.ndim)
        return np.mean(x)


if __name__ == "__main__":

    arr = []
    arr.append(np.array([[0.3148, 0.0478, 0.6243, 0.4608],
                  [0.7149, 0.0775, 0.6072, 0.9656],
                  [0.6341, 0.1403, 0.9759, 0.4064],
                  [0.5918, 0.6948, 0.904, 0.3721],
                  [0.0921, 0.2481, 0.1188, 0.1366]]))
    arr.append(np.array(0.1366))
    arr.append(np.array([0.1, 0.2]))
    arr.append(np.array([[0.1], [0.2]]))
    arr.append(np.array([[0.1, 0.2]]))
    arr.append(np.array([[[0.1], [0.2]]]))
    arr.append(np.array([[[0.1, 1], [0.2, 2]]]))

    for a in arr :
        print(zscore(a))
