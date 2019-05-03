import pandas as pd
from reco_engine.prepare_data import read_movie_ratings
import sys
from sklearn.metrics.pairwise import cosine_similarity
from reco_engine.Contents import Content

class Base_Recommender():

    def __init__(self):
        self.model_key = "None"

    def get_data(self):
        ratings = pd.read_csv(r'C:\datasets\the-movies-dataset\prep_ratings.csv')
        return ratings

    def get_model(self, feature_list, model_type):
        return pd.read_csv(
            r"C:\datasets\the-movies-dataset\models\collaborative\content_" + self.model_key + "_"+model_type+".csv")


class CosSim_Recommender(Base_Recommender):
    def __init__(self, type):
        Base_Recommender.__init__(self)
        self.model_key = type

    def create_model(self):
        n=100
        raw_data = self.get_data()
        if self.model_key == "movie":
            pv_index="movieId"
            pv_columns="userId"
        else:
            pv_index = "userId"
            pv_columns = "movieId"
        coll_matrix=raw_data[:n].pivot_table(values='rating', index=pv_index, columns=pv_columns).fillna(0)
        # print(tfidf_matrix.shape, cosine_sim.shape)
        # print(cosine_sim.shape)
        print(coll_matrix)
        cosine_sim = cosine_similarity(coll_matrix, coll_matrix)
        print(cosine_sim.shape)
        #print(cosine_sim)


        if self.model_key == "movie":
            ele_lst = self.get_data()["id"].values.tolist()
        else:
            ele_lst = self.get_data()["userId"].values.tolist()

        print(len(ele_lst))
        print(cosine_sim.shape)

        pd.DataFrame(data=cosine_sim, index=ele_lst, columns=ele_lst).to_csv(
            "C:/datasets/the-movies-dataset/models/collaborative_based/content_" + self.model_key + ".csv")

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #ratings = read_movie_ratings()
    CS_Rec = CosSim_Recommender("movie")
    CS_Rec.create_model()