from sklearn.base import BaseEstimator, ClassifierMixin
from reco_engine import Config
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.externals import joblib


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
        print(pv_data)
        #adjust ratings to evite the user tendence like hihg or low rating
        pv_data = pv_data-pv_data.mean()
        print(pv_data)
        #add mock rating data based on means of each user to recommend never watched contents
        pv_data.loc[-1] = [pv_data[c].mean() for c in pv_data.columns.values.tolist()]
        pv_data.fillna(0, inplace=True)
        sim_matrix = cosine_similarity(pv_data, pv_data)
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
    import time
    ratings = pd.read_csv(r'C:\datasets\the-movies-dataset\prep_ratings.csv')
    CCR = CosineContentR()

    CCR.fit(ratings)
    #joblib.dump(CCR, "test.sav")
    #CCR = None
    #CCR2 = joblib.load("test.sav")
    #print(CCR2.predict(197))
