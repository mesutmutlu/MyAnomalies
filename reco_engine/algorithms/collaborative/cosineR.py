from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from reco_engine import Config
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.externals import joblib
import numpy as np


class CosineR(BaseEstimator, RegressorMixin):

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

        pv_data_implicit = pv_data_adjusted.notnull().astype(int)

        pv_data_adjusted.fillna(0, inplace=True)
        sim_matrix = cosine_similarity(pv_data_adjusted, pv_data_adjusted)

        nominator = np.dot(sim_matrix, pv_data_adjusted)
        denominator = np.absolute(np.dot(sim_matrix,pv_data_implicit))

        p_ratings = np.add(mu.values.reshape(-1,1),(nominator / denominator))
        self.predictions = sparse.csr_matrix(p_ratings)
        self.similarity_matrix = sparse.csr_matrix(sim_matrix)

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
            raise SystemExit('You should fit your estimator or load the model before make a prediction')

        pre_ratings = np.zeros((len(x),))

        model = pd.DataFrame(self.predictions.todense(), index=self.rows, columns=self.columns)
        for i in range(len(x)):
            if (x[i, 0]== np.NaN) or (x[i, 1]== np.NaN):
                raise SystemExit('X should not have NaN value')

            if x[i, 0] not in self.rows.tolist():
                x[i, 0] = -1
            if x[i, 1] not in self.columns.tolist():
                raise SystemExit(x[i, 1] + ' not in model')
            pre_ratings[i] = model.loc[x[i,0], x[i,1]]

        return pre_ratings

    def get_similars(self, x0_id, n=10):

        if self.similarity_matrix is None:
            raise SystemExit('You should fit your estimator or load the model to get similarities')
        if x0_id == None:
            raise SystemExit('x0_id should be defined')
        if x0_id not in self.rows.tolist():
            x0_id = -1

        similarities = pd.DataFrame(self.similarity_matrix.todense(), index=self.rows, columns=self.rows).loc[x0_id].drop(x0_id)
        return similarities.sort_values(ascending=False)[:n]

    #def score(self, x=None, y=None):
        #pass


class CosineUserR(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.pv_index = Config.Attributes.c_userId
        self.pv_columns = Config.Attributes.c_contentId
        self.pv_values = Config.Attributes.c_rating
        self.model = None
        self.model_keys = None

    def fit(self, data, pv_index=None, pv_columns=None, pv_values=None):
        if pv_index is not None:
            self.pv_index = pv_index
        if pv_columns is not None:
            self.pv_columns = pv_columns
        if pv_values is not None:
            self.pv_values = pv_values

        pv_data = data.pivot_table(values=self.pv_values, index=self.pv_index, columns=self.pv_columns).fillna(0)

        sim_matrix = cosine_similarity(pv_data, pv_data)
        self.model = sparse.csr_matrix(sim_matrix)
        self.model_keys = pv_data.index.tolist()

    def predict(self, userId, n=10):
        print("Finding similar movies for ", idx, Usr.user["username"], "using", self.model_key)
        model = self.get_model()
        if str(idx) not in model.columns.values.tolist():
            return pd.DataFrame(columns=["title", "pre_rating"])
        user_sim = model[[str(idx)]].rename(columns={str(idx): 'similarity'})
        user_sim = user_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
        users = [int(id) for id in user_sim.index.values.tolist()]

        ratings = self.get_ratings().pivot_table(values="rating", columns=self.pv_index, index=self.pv_columns)[users] \
            .apply(lambda row: row.fillna(row.sum() / n), axis=1)
        mul1 = np.dot(ratings, user_sim)
        denom1 = user_sim.sum().values
        predicted_ratings = pd.DataFrame(data=(mul1 / denom1) + (Usr.get_rating_history()["rating"].std()),
                                         index=ratings.index.values.tolist(), columns=["pre_rating"])
        print(predicted_ratings)
        predicted_ratings.index.name = "id"
        print(len(predicted_ratings), len(predicted_ratings.drop(labels=Usr.get_rating_history().index, axis=0)))
        if mode == "live":
            predicted_ratings.drop(labels=Usr.get_rating_history().index, inplace=True, axis=0)
        predicted_ratings = predicted_ratings.join(Content_Helper.get_content_list()["title"], how="left")
        return predicted_ratings.sort_values(by=["pre_rating"], ascending=False)[:n].reset_index()


if __name__ == "__main__":

    import pandas as pd
    import sys
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    a = np.array(["u4", "i4"])

    v = np.array([["u1", "i3", 2],
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

    #print(v[:,0:2])
    #sys.exit()
    data = pd.DataFrame(data=v, columns=["userId", "id", "rating"])
    # b1 = pd.DataFrame(data = [[1,2],[1,10]])
    # print(b1)
    # print(b1.mean(axis=1))
    # print(b1-b1.mean(axis=1))
    CCR = CosineR(c_index="userId", c_columns="id")
    CCR.fit(v[:,0:2], v[:,2])
    print(CCR.predict(np.array([["u4", "i4"],["u5", "i1"]])))
    print(CCR.score(v[:,0:2], v[:,2].astype(float).ravel()))




    import time
    ratings = pd.read_csv(r'C:\datasets\the-movies-dataset\prep_ratings.csv')

    # pv_data = ratings.pivot_table(values="rating", index="id", columns="userId")
    # #print(pv_data.mean(axis=1).reshape(-1,1))
    # print(pv_data)
    # print(pv_data.mean(axis=1))
    # print(pv_data.subtract(pv_data.mean(axis=1), axis=0))
    #sys.exit()
    CCR = CosineR(c_index="userId", c_columns="id")
    CCR.fit(ratings[["userId", "id"]], ratings["rating"])
    print(CCR.score(ratings[["userId", "id"]],ratings["rating"]))
    print(CCR.predict(ratings[["userId", "id"]]))
    #print(CCR.get_similars(11, 3))

    #CCR.fit(data,pv_index="id", pv_columns="userId")
    #joblib.dump(CCR, "test.sav")
    #CCR = None
    #CCR2 = joblib.load("test.sav")
    #print(CCR2.predict(197))
