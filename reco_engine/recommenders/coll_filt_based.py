import pandas as pd
from reco_engine.prepare_data import read_movie_ratings
import sys
from sklearn.metrics.pairwise import cosine_similarity
from reco_engine.Contents import Content
from reco_engine.Users import User
import numpy as np

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
        #print(raw_data.head())
        #print(raw_data.shape)
        #print(raw_data[raw_data["id"]<0])
        #print(raw_data[raw_data["userId"] < 0])
        #print(len(raw_data.userId.unique()), len(raw_data.id.unique()))

        coll_matrix=raw_data.pivot_table(values='rating', index=self.pv_index, columns=self.pv_columns).fillna(0)
        # print(tfidf_matrix.shape, cosine_sim.shape)
        # print(cosine_sim.shape)
        #print(coll_matrix.shape)
        #print(coll_matrix)
        cosine_sim = cosine_similarity(coll_matrix, coll_matrix)
        #print("cosine shape", cosine_sim.shape)
        #print(cosine_sim)

        ele_lst = [str(int(x)) for x in  raw_data[self.pv_index ].unique()]

        #print("ele_slt", len(ele_lst))
        #print(ele_lst)

        pd.DataFrame(data=cosine_sim, index=ele_lst, columns=ele_lst).to_csv(
            "C:/datasets/the-movies-dataset/models/collaborative_based/coll_" + self.model_key + ".csv")

    def make_recommendation_by_movie(self,title, n):
        # Obtain the id of the movie that matches the title
        Cnt = Content()
        Cnt.load_content_list()
        Cnt.movielist.set_index("title", inplace=True)
        #print(Cnt.movielist)
        idx = Cnt.get_id_by_title(title)
        print("Finding similar movies based on ", idx, title, "using", self.model_key)
        movie_sim = self.get_model()[[str(idx)]].rename(columns={str(idx): 'similarity'})
        movie_sim = movie_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
        Cnt.movielist.reset_index().set_index("id", inplace=True)
        return pd.concat([Cnt.get_contents_by_id_list(movie_sim.index.values.tolist()), movie_sim], axis=1)

    def make_recommendation_by_user(self,user_name, n):
        # Obtain the id of the movie that matches the title
        Usr = User()
        idx = Usr.get_userid_by_name(user_name)
        print("Finding similar movies for ", idx, user_name, "using", self.model_key)
        user_sim = self.get_model()[[str(idx)]].rename(columns={str(idx): 'similarity'})
        user_sim = user_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
        Usr.users.reset_index().set_index("userid", inplace=True)
        print("similar_users")
        print(pd.concat([Usr.get_users_by_id_list(user_sim.index.values.tolist()), user_sim], axis=1))
        ratings = self.get_data().pivot_table(values="rating", columns=self.pv_index, index=self.pv_columns).fillna(0)[user_sim.index.values.tolist()]
        print(ratings.loc[862,user_sim.index.values.tolist()])
        #print(ratings)
        predicted_ratings = pd.DataFrame(data= np.dot(ratings, user_sim), index = ratings.index.values.tolist(), columns=["pre_rating"])
        predicted_ratings.index.name = "id"
        print(predicted_ratings)
        Cnt = Content()
        Cnt.load_content_list()
        print(Cnt.movielist.set_index("id"))
        predicted_ratings= predicted_ratings.join(Cnt.movielist.set_index("id")["title"], how="left")
        print(predicted_ratings.sort_values(by=["pre_rating"], ascending=False))
        print(self.get_data()[self.get_data()["userId"]==11].sort_values(by=["rating"], ascending=False))
        return predicted_ratings.sort_values(by=["pre_rating"], ascending=False)



if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #ratings = read_movie_ratings()
    CS_Rec = CosSim_Recommender("user")
    CS_Rec.create_model()
    #print(CS_Rec.get_model().shape)
    #print(CS_Rec.get_model())
    print(CS_Rec.make_recommendation_by_user("Julie Wallis", 10))