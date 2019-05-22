from sklearn.base import BaseEstimator, ClassifierMixin
from reco_engine import Config
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.externals import joblib
import numpy as np


class CosineContentR(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.pv_index = Config.Attributes.c_contentId
        self.pv_columns = Config.Attributes.c_userId
        self.pv_values = Config.Attributes.c_rating
        self.model = None

    def fit(self, data, pv_index=None, pv_columns=None, pv_values=None):
        if pv_index is not None:
            self.pv_index = pv_index
        if pv_columns is not None:
            self.pv_columns = pv_columns
        if pv_values is not None:
            self.pv_values = pv_values
        pv_data = data.pivot_table(values=self.pv_values, index=self.pv_index, columns=self.pv_columns)
        #print(pv_data)
        mu = pv_data.mean(axis=1)
        mu.loc[-1] = mu.mean()
        #print(mu)
        #adjust ratings to evite the user tendence like hihg or low rating

        pv_data_adjusted = pv_data.subtract(pv_data.mean(axis=1), axis=0)
        print(pv_data[0:1].sum(axis=1))
        print(pv_data[0:1].mean(axis=1))
        print(pv_data.shape, mu.shape, pv_data_adjusted.shape)
        # add mock rating data based on means of each user to recommend never watched contents

        pv_data_adjusted.loc[-1] = [pv_data_adjusted[c].mean() for c in pv_data_adjusted.columns.values.tolist()]
        pv_data_implicit = pv_data_adjusted.notnull().astype(int)

        #print(pv_data_adjusted)
        #print(pv_data_implicit)
        pv_data_adjusted.fillna(0, inplace=True)
        sim_matrix = cosine_similarity(pv_data_adjusted, pv_data_adjusted)
        print(mu.shape, sim_matrix.shape, pv_data_adjusted.shape, pv_data_implicit.shape)
        print("sim matrix")
        print(np.sum(sim_matrix[0:1]))
        print("data adjusted")
        print(pv_data_adjusted.loc[:,1])

        nominator = np.dot(sim_matrix, pv_data_adjusted)
        print("nominator")
        print(nominator)
        denominator = np.absolute(np.dot(sim_matrix,pv_data_implicit))
        print("denominator")
        print(denominator)
        #denominator[denominator == 0] = 1
        self.model = np.add(mu.values.reshape(-1,1),(nominator / denominator))
        #print(self.model)
        self.model = sparse.csr_matrix(sim_matrix)
        self.model_keys = pv_data.index.tolist()
        #self.model=pd.DataFrame(data=sim_matrix, index=pv_data.index.tolist(), columns=pv_data.index.tolist())

    def predict(self, contentId, n=10):
        print("Finding similar movies based on ", contentId)
        if self.model is None:
            raise SystemExit('You should fit your estimator or load the model before make a prediction')
        if contentId not in self.model_keys:
            print('Content requested has not any watch history recommendation through mock data')
            contentId = -1

        pd_model = pd.DataFrame(self.model.todense(), index=self.model_keys, columns=self.model_keys).drop(contentId)
        sim_movies = pd_model[[contentId]].rename(columns={contentId: 'similarity'})
        sim_movies.index.name = self.pv_index
        print(pd_model)
        sim_movies = sim_movies.sort_values(by=["similarity"], ascending=False)[1:n + 1]

        return sim_movies

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
    v = [["u1", "i3", 2],
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
         ["u7", "i1", 1],
         ["u7", "i3", 3],
         ]
    data = pd.DataFrame(data=)
    # b1 = pd.DataFrame(data = [[1,2],[1,10]])
    # print(b1)
    # print(b1.mean(axis=1))
    # print(b1-b1.mean(axis=1))




    import time
    ratings = pd.read_csv(r'C:\datasets\the-movies-dataset\prep_ratings.csv')

    # pv_data = ratings.pivot_table(values="rating", index="id", columns="userId")
    # #print(pv_data.mean(axis=1).reshape(-1,1))
    # print(pv_data)
    # print(pv_data.mean(axis=1))
    # print(pv_data.subtract(pv_data.mean(axis=1), axis=0))
    #sys.exit()
    CCR = CosineContentR()

    CCR.fit(ratings,pv_index="id", pv_columns="userId")
    #joblib.dump(CCR, "test.sav")
    #CCR = None
    #CCR2 = joblib.load("test.sav")
    #print(CCR2.predict(197))
