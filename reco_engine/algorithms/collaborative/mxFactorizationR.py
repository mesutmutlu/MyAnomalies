from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from reco_engine import Config
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import jaccard_similarity_score
from scipy import sparse
from sklearn.externals import joblib
import numpy as np
from scipy.spatial.distance import cosine
from scipy import linalg
import pandas as pd
from scipy.misc import derivative



class SvdR(BaseEstimator, RegressorMixin):

    def __init__(self, c_index=None, c_columns=None):
        self.pv_index = c_index
        self.pv_columns = c_columns
        self.predictions = None
        self.rows = None
        self.columns = None

    def fit(self, x=None, y=None):
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = x.values

        if (type(y) == pd.DataFrame) or (type(y) == pd.Series):
            y = y.values.reshape(-1,1)

        if type(y) == list:
            y = np.array(y).reshape(-1,1).ravel()

        if  y.ndim != 1:
            y = y.ravel()

        if x.shape[1] != 2:
            raise SystemExit("x should have 2 columns, first as values for which the similarity will be calculated")

        if y.ndim != 1:
            raise SystemExit("y should have 1 column")

        if len(x) != len(y):
            raise SystemExit("x and y should have same length")

        rows, row_pos = np.unique(x[:, 0], return_inverse=True)
        cols, col_pos = np.unique(x[:, 1], return_inverse=True)

        pivot_table = np.zeros((len(rows), len(cols)), dtype=np.float)
        pivot_table[row_pos, col_pos] = y

        pv_data = pd.DataFrame(data=pivot_table, index=rows, columns=cols)
        pv_data.replace(0, np.NaN, inplace=True)

        # add mock values data based on means of each column(ex for items for user/item matrix
        pv_data.loc[-1] = pv_data.mean(axis=0).values.T
        mu = pv_data.mean(axis=1)

        #adjust values to evite the user tendence like high or low rating
        pv_data_adjusted = pv_data.subtract(mu, axis=0) + 0.1 #0.1 as regularization factor
        pv_data_adjusted.fillna(0, inplace=True)
        #print(pv_data_adjusted)

        #pv_data_adjusted.fillna(0, inplace=True)
        U, s, Vh = linalg.svd(pv_data_adjusted.values, full_matrices=False)

        self.predictions = sparse.csr_matrix(np.add(np.dot(U, np.dot(np.diag(s), Vh)), mu.values.reshape(-1,1)))
        self.rows = np.append(rows, [-1])
        self.columns = cols

    def predict(self, x=None):
        if type(x) == pd.DataFrame:
            x = x.values
        if type(x) != type(np.empty((1,2))):
            raise SystemExit("You should enter a numpy array of shape (n,2) as input")
        if x.ndim != 2:
            raise SystemExit("You should enter a numpy array of shape (n,2) as input")
        if self.predictions is None:
            raise SystemExit('You should fit your estimator or load a fitted model before make a prediction')

        pre_ratings = np.zeros((len(x),))

        model = pd.DataFrame(self.predictions.todense(), index=self.rows, columns=self.columns)
        for i in range(len(x)):
            if (x[i, 0] == np.NaN) or (x[i, 1] == np.NaN):
                raise SystemExit('X should not have NaN value')
            if x[i, 0] not in self.rows.tolist():
                x[i, 0] = -1
            if x[i, 1] not in self.columns.tolist():
                raise SystemExit(x[i, 1] + ' not in model as to be predicted')
            pre_ratings[i] = model.loc[x[i, 0], x[i, 1]]

        return pre_ratings


class FunkSvdR(BaseEstimator, RegressorMixin):

    def __init__(self, c_index=None, c_columns=None):
        self.pv_index = c_index
        self.pv_columns = c_columns
        self.predictions = None
        self.rows = None
        self.columns = None

    def fit(self, x=None, y=None, latent_features=12, learning_rate=0.001, iters=1000, early_stop=0.1,
            regularization=0.01, randomstate=42):
        np.random.seed(randomstate)
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = x.values

        if (type(y) == pd.DataFrame) or (type(y) == pd.Series):
            y = y.values.reshape(-1,1)

        if type(y) == list:
            y = np.array(y).reshape(-1,1).ravel()

        if y.ndim != 1:
            y = y.ravel()

        if x.shape[1] != 2:
            raise SystemExit("x should have 2 columns, first as values for which the similarity will be calculated")

        if y.ndim != 1:
            raise SystemExit("y should have 1 column")

        if len(x) != len(y):
            raise SystemExit("x and y should have same length")

        rows, row_pos = np.unique(x[:, 0], return_inverse=True)
        cols, col_pos = np.unique(x[:, 1], return_inverse=True)

        pivot_table = np.zeros((len(rows), len(cols)), dtype=np.float)
        pivot_table[row_pos, col_pos] = y

        pv_data = pd.DataFrame(data=pivot_table, index=rows, columns=cols)
        pv_data.replace(0, np.NaN, inplace=True)

        # add mock values data based on means of each column(ex for items for user/item matrix
        pv_data.loc[-1] = pv_data.mean(axis=0).values.T

        #adjust values to evite the user tendence like high or low rating
        pv_data_adjusted = pv_data.subtract(pv_data.mean(axis=1), axis=0) + 0.1 #0.1 as regularization factor
        #print(pv_data_adjusted)
        pv_data_adjusted = pv_data_adjusted.values

        self.rows = np.append(rows, [-1])
        self.columns = cols

        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Set up some useful values for later
        self.n_users = len(self.rows)
        self.n_items = len(self.columns)
        self.num_ratings = np.count_nonzero(~np.isnan(pv_data))

        self.user_mat = np.random.rand(self.n_users, self.latent_features)
        self.item_mat = np.random.rand(self.latent_features, self.n_items)
        self.bu = np.random.rand(self.n_users, 1)
        self.bi = np.random.rand(self.n_items, 1)
        self.mu = pv_data.mean().values

        sse_accum = 0

        print("Iterations \t\t Mean Squared Error ")
        print(pv_data)
        for iteration in range(self.iters):
            old_sse = sse_accum
            sse_accum = 0
            #print(iteration, old_sse, self.num_ratings, old_sse/self.num_ratings)
            if (iteration != 0) & (old_sse/self.num_ratings < 0.1):
                break
            for i in range(self.n_users):
                for j in range(self.n_items):

                    # if the rating exists (so we train only on non-missval)
                    if pv_data_adjusted[i, j] > 0:
                        # compute the error as the actual minus the dot
                        # product of the user and item latent features
                        diff = (
                                pv_data_adjusted[i, j]
                                - np.dot(self.user_mat[i, :], self.item_mat[:, j])
                        )
                        # Keep track of the sum of squared errors for the
                        # matrix
                        sse_accum += diff ** 2
                        self.bu[i] += self.learning_rate * (diff - regularization * self.bu[i])
                        self.bi[j] += self.learning_rate * (diff - regularization * self.bu[j])

                        for k in range(self.latent_features):
                            self.user_mat[i, k] += self.learning_rate * \
                                                   (diff * self.item_mat[k, j] - regularization * self.user_mat[i, k])

                            self.item_mat[k, j] += self.learning_rate * \
                                                   (diff * self.user_mat[i, k] - regularization * self.item_mat[k, j])

            print(f"\t{iteration+1} \t\t {sse_accum/self.num_ratings} ")

        self.predictions = sparse.csr_matrix(np.dot(self.user_mat, self.item_mat) + self.mu + self.bu +self.bi.reshape(1,len(self.bi)))
        self.rows = np.append(rows, [-1])
        self.columns = cols


    def predict(self, x=None):
        if type(x) == pd.DataFrame:
            x = x.values
        if type(x) != type(np.empty((1,2))):
            raise SystemExit("You should enter a numpy array of shape (n,2) as input")
        if x.ndim != 2:
            raise SystemExit("You should enter a numpy array of shape (n,2) as input")
        if self.predictions is None:
            raise SystemExit('You should fit your estimator or load a fitted model before make a prediction')

        pre_ratings = np.zeros((len(x),))

        model = pd.DataFrame(self.predictions.todense(), index=self.rows, columns=self.columns)
        for i in range(len(x)):
            if (x[i, 0] == np.NaN) or (x[i, 1] == np.NaN):
                raise SystemExit('X should not have NaN value')
            if x[i, 0] not in self.rows.tolist():
                x[i, 0] = -1
            if x[i, 1] not in self.columns.tolist():
                raise SystemExit(x[i, 1] + ' not in model as to be predicted')
            pre_ratings[i] = model.loc[x[i, 0], x[i, 1]]

        return pre_ratings

class SvdPPR(BaseEstimator, RegressorMixin):

    def __init__(self, c_index=None, c_columns=None):
        self.pv_index = c_index
        self.pv_columns = c_columns
        self.predictions = None
        self.rows = None
        self.columns = None

    def fit(self, x=None, y=None):
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = x.values

        if (type(y) == pd.DataFrame) or (type(y) == pd.Series):
            y = y.values.reshape(-1,1)

        if type(y) == list:
            y = np.array(y).reshape(-1,1).ravel()

        if  y.ndim != 1:
            y = y.ravel()

        if x.shape[1] != 2:
            raise SystemExit("x should have 2 columns, first as values for which the similarity will be calculated")

        if y.ndim != 1:
            raise SystemExit("y should have 1 column")

        if len(x) != len(y):
            raise SystemExit("x and y should have same length")

        rows, row_pos = np.unique(x[:, 0], return_inverse=True)
        cols, col_pos = np.unique(x[:, 1], return_inverse=True)

        pivot_table = np.zeros((len(rows), len(cols)), dtype=np.float)
        pivot_table[row_pos, col_pos] = y

        pv_data = pd.DataFrame(data=pivot_table, index=rows, columns=cols)
        pv_data.replace(0, np.NaN, inplace=True)

        # add mock values data based on means of each column(ex for items for user/item matrix
        pv_data.loc[-1] = pv_data.mean(axis=0).values.T
        mu = pv_data.mean(axis=1)

        #adjust values to evite the user tendence like high or low rating
        pv_data_adjusted = pv_data.subtract(mu, axis=0) + 0.1 #0.1 as regularization factor
        pv_data_adjusted.fillna(0, inplace=True)
        #print(pv_data_adjusted)

        #pv_data_adjusted.fillna(0, inplace=True)
        U, s, Vh = linalg.svd(pv_data_adjusted.values, full_matrices=False)

        self.predictions = sparse.csr_matrix(np.add(np.dot(U, np.dot(np.diag(s), Vh)), mu.values.reshape(-1,1)))
        self.rows = np.append(rows, [-1])
        self.columns = cols

    def predict(self, x=None):
        if type(x) == pd.DataFrame:
            x = x.values
        if type(x) != type(np.empty((1,2))):
            raise SystemExit("You should enter a numpy array of shape (n,2) as input")
        if x.ndim != 2:
            raise SystemExit("You should enter a numpy array of shape (n,2) as input")
        if self.predictions is None:
            raise SystemExit('You should fit your estimator or load a fitted model before make a prediction')

        pre_ratings = np.zeros((len(x),))

        model = pd.DataFrame(self.predictions.todense(), index=self.rows, columns=self.columns)
        for i in range(len(x)):
            if (x[i, 0] == np.NaN) or (x[i, 1] == np.NaN):
                raise SystemExit('X should not have NaN value')
            if x[i, 0] not in self.rows.tolist():
                x[i, 0] = -1
            if x[i, 1] not in self.columns.tolist():
                raise SystemExit(x[i, 1] + ' not in model as to be predicted')
            pre_ratings[i] = model.loc[x[i, 0], x[i, 1]]

        return pre_ratings


class HybridSvdR(BaseEstimator, RegressorMixin):

    def __init__(self, c_index=None, c_columns=None):
        self.pv_index = c_index
        self.pv_columns = c_columns
        self.predictions = None
        self.rows = None
        self.columns = None

    def fit(self, x=None, y=None):
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = x.values

        if (type(y) == pd.DataFrame) or (type(y) == pd.Series):
            y = y.values.reshape(-1,1)

        if type(y) == list:
            y = np.array(y).reshape(-1,1).ravel()

        if  y.ndim != 1:
            y = y.ravel()

        if x.shape[1] != 2:
            raise SystemExit("x should have 2 columns, first as values for which the similarity will be calculated")

        if y.ndim != 1:
            raise SystemExit("y should have 1 column")

        if len(x) != len(y):
            raise SystemExit("x and y should have same length")

        rows, row_pos = np.unique(x[:, 0], return_inverse=True)
        cols, col_pos = np.unique(x[:, 1], return_inverse=True)

        pivot_table = np.zeros((len(rows), len(cols)), dtype=np.float)
        pivot_table[row_pos, col_pos] = y

        pv_data = pd.DataFrame(data=pivot_table, index=rows, columns=cols)
        pv_data.replace(0, np.NaN, inplace=True)

        # add mock values data based on means of each column(ex for items for user/item matrix
        pv_data.loc[-1] = pv_data.mean(axis=0).values.T
        mu = pv_data.mean(axis=1)

        #adjust values to evite the user tendence like high or low rating
        pv_data_adjusted = pv_data.subtract(mu, axis=0) + 0.1 #0.1 as regularization factor
        pv_data_adjusted.fillna(0, inplace=True)
        #print(pv_data_adjusted)

        #pv_data_adjusted.fillna(0, inplace=True)
        U, s, Vh = linalg.svd(pv_data_adjusted.values, full_matrices=False)

        self.predictions = sparse.csr_matrix(np.add(np.dot(U, np.dot(np.diag(s), Vh)), mu.values.reshape(-1,1)))
        self.rows = np.append(rows, [-1])
        self.columns = cols

    def predict(self, x=None):
        if type(x) == pd.DataFrame:
            x = x.values
        if type(x) != type(np.empty((1,2))):
            raise SystemExit("You should enter a numpy array of shape (n,2) as input")
        if x.ndim != 2:
            raise SystemExit("You should enter a numpy array of shape (n,2) as input")
        if self.predictions is None:
            raise SystemExit('You should fit your estimator or load a fitted model before make a prediction')

        pre_ratings = np.zeros((len(x),))

        model = pd.DataFrame(self.predictions.todense(), index=self.rows, columns=self.columns)
        for i in range(len(x)):
            if (x[i, 0] == np.NaN) or (x[i, 1] == np.NaN):
                raise SystemExit('X should not have NaN value')
            if x[i, 0] not in self.rows.tolist():
                x[i, 0] = -1
            if x[i, 1] not in self.columns.tolist():
                raise SystemExit(x[i, 1] + ' not in model as to be predicted')
            pre_ratings[i] = model.loc[x[i, 0], x[i, 1]]

        return pre_ratings



def sgd(f, v):
    pass
if __name__ == "__main__":
    import sys

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    a = np.array(["u4", "i4"])

    v = np.array([["u1", "i2", 2],
                  ["u1", "i3", 3],
                  ["u2", "i2", 3],
                  ["u2", "i3", 3],
                  ["u2", "i4", 1],
                  ["u2", "i5", 4],
                  ["u2", "i6", 4],
                  ["u3", "i3", 5],
                  ["u3", "i4", 3],
                  ["u3", "i5", 3],
                  ["u4", "i2", 5],
                  ["u4", "i4", 5],
                  ["u5", "i1", 2.5],
                  ["u5", "i2", 4],
                  ["u5", "i3", 2],
                  ["u5", "i6", 4],
                  ["u6", "i4", 2],
                  ["u7", "i2", 1],
                  ["u7", "i5", 3],
                  ])

    # print(v[:,0:2])
    # sys.exit()
    data = pd.DataFrame(data=v, columns=["userId", "id", "rating"])
    # b1 = pd.DataFrame(data = [[1,2],[1,10]])
    # print(b1)
    # print(b1.mean(axis=1))
    # print(b1-b1.mean(axis=1))
    svd = SvdR(c_index="userId", c_columns="id")
    fsvd = FunkSvdR(c_index="userId", c_columns="id")
    fsvd.fit(v[:, 0:2], v[:, 2])

    print("user_mat")
    print(fsvd.user_mat)
    print("item_mat")
    print(fsvd.item_mat)
    print("mu")
    print(fsvd.mu)
    print("bu")
    print(fsvd.bu)
    print("bi")
    print(fsvd.bi)

    print(fsvd.predict(v[:, 0:2]))
    print(v[:, 2])