import pandas as pd
from reco_engine.prepare_data import read_movie_ratings
import sys
from sklearn.metrics.pairwise import cosine_similarity
from reco_engine.Lib_Content import Content, Content_Helper
from reco_engine.Lib_User import User
import numpy as np
from surprise import SVD, SVDpp, Dataset, Reader, evaluate


class Base_Recommender():

    def __init__(self, type="movie"):
        self.model_key = type
        if type == "movie":
            self.pv_index = "id"
            self.pv_columns = "userId"
        else:
            self.pv_index = "userId"
            self.pv_columns = "id"

    def get_data(self):
        ratings = pd.read_csv(r'C:\datasets\the-movies-dataset\prep_ratings.csv')
        return ratings

    def get_model(self):
        return pd.read_csv(
            r"C:/datasets/the-movies-dataset/models/collaborative_based/coll_" + self.model_key + ".csv", index_col=0)


class CosSim_Recommender(Base_Recommender):
    def __init__(self, type):
        Base_Recommender.__init__(self, type)

    def create_model(self):
        n=61000
        raw_data = self.get_data()[:n].fillna(0)
        coll_matrix=raw_data.pivot_table(values='rating', index=self.pv_index, columns=self.pv_columns).fillna(0)
        cosine_sim = cosine_similarity(coll_matrix, coll_matrix)

        ele_lst = [str(int(x)) for x in  raw_data[self.pv_index ].unique()]
        pd.DataFrame(data=cosine_sim, index=ele_lst, columns=ele_lst).to_csv(
            "C:/datasets/the-movies-dataset/models/collaborative_based/coll_" + self.model_key + ".csv")

    def make_recommendation_by_movie(self, idx, n):
        # Obtain the id of the movie that matches the title
        Cnt = Content(idx)
        print("Finding similar movies based on ", idx, Cnt.content["title"], "using", self.model_key)
        model = self.get_model()
        if str(idx) in model.columns.values.tolist():
            movie_sim = model[[str(idx)]].rename(columns={str(idx): 'similarity'})
            movie_sim = movie_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
            return pd.concat([Content_Helper.get_contents_by_id_list(movie_sim.index.values.tolist())[["title"]], movie_sim], axis=1)[:n]
        else:
            return pd.DataFrame(columns=["title", "similarity"])

    def make_recommendation_by_user(self, idx, n):
        # Obtain the id of the movie that matches the title
        Usr = User(idx)
        print("Finding similar movies for ", idx, Usr.user["username"], "using", self.model_key)
        model = self.get_model()
        if str(idx) not in model.columns.values.tolist():
            return pd.DataFrame(columns=["title", "pre_rating"])
        user_sim = model[[str(idx)]].rename(columns={str(idx): 'similarity'})
        user_sim = user_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
        #print("similar_users")
        #print(pd.concat([Usr.get_users_by_id_list(user_sim.index.values.tolist()), user_sim], axis=1))
        ratings = self.get_data().pivot_table(values="rating", columns=self.pv_index, index=self.pv_columns).fillna(0)[user_sim.index.values.tolist()]
        #print(ratings.loc[862,user_sim.index.values.tolist()])
        #print(ratings)
        predicted_ratings = pd.DataFrame(data= np.dot(ratings, user_sim), index = ratings.index.values.tolist(), columns=["pre_rating"])
        predicted_ratings.index.name = "id"
        predicted_ratings.drop(labels=Usr.get_rating_history().index,inplace=True, axis=0)
        #print(predicted_ratings)
        #print(Cnt.movielist.set_index("id"))
        predicted_ratings= predicted_ratings.join(Content_Helper.get_content_list()["title"], how="left")
        #print(predicted_ratings.sort_values(by=["pre_rating"], ascending=False))
        #print(self.get_data()[self.get_data()["userId"]==11].sort_values(by=["rating"], ascending=False))
        return predicted_ratings.sort_values(by=["pre_rating"], ascending=False)[:n].reset_index()

class SVD_Recommender(Base_Recommender):
    def __init__(self, type):
        Base_Recommender.__init__(self, type)

    def create_model(self):
        n = 61000

        raw_data = self.get_data()[:n].fillna(0)[["userId", "id", "rating"]]
        reader = Reader()
        data = Dataset.load_from_df(raw_data, reader)
        data.split(n_folds=5)
        svdpp = SVDpp()
        svdpp.fit(data)
        trainset = data.build_full_trainset()
        evaluate(svdpp, data, measures=['RMSE'])


if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    SVD_Rec = SVD_Recommender("user")
    SVD_Rec.create_model()
    #from surprise.model_selection import train_test_split


    # Load the movielens-100k dataset (download it if needed),
    #data = Dataset.load_builtin('ml-100k')
    #trainset, testset = train_test_split(data, test_size=.25)
    #print(trainset)
    #ratings = read_movie_ratings()
    #CS_Rec = CosSim_Recommender("movie")
    #CS_Rec.create_model()
    #CS_Rec = CosSim_Recommender("user")
    #CS_Rec.create_model()
    #print(CS_Rec.get_model().shape)
    #print(CS_Rec.get_model())
    #print(CS_Rec.make_recommendation_by_user(22, 10))

