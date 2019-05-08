import pandas as pd
import numpy as np
from reco_engine.Contents import Content
import sys
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.neighbors import NearestNeighbors, BallTree
import pickle
from sklearn.decomposition import TruncatedSVD


class Base_Recommender():

    def __init__(self):
        self.all_features = ["overview", "cast", "keywords", "leads", "genres", "belongs_to_collection"]
        self.model_key = "_".join(self.all_features)

    def get_data(self):
        self.raw_data = pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")
        return pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")

    def get_model(self, feature_list, model_type):
        self.set_model_key(feature_list)
        return pd.read_csv(
            r"C:\datasets\the-movies-dataset\models\content_based\content_" + self.model_key + "_"+model_type+".csv")


    def set_model_key(self, feature_list):
        if len(feature_list) == 1:
            self.model_key = feature_list[0]
        elif type(feature_list) == str:
            self.model_key = feature_list
        else:
            self.model_key = "_".join(feature_list)

class CosSim_Recommender(Base_Recommender):

    def __init__(self):
        Base_Recommender.__init__(self)
        self.model_type = "cossim"

    def create_model(self, feature_list):
        n = 10000
        movies_df = self.get_data()
        df_cossim = movies_df[feature_list].fillna("").apply(lambda x: ''.join(x), axis=1)
        #print(movies_df)
        #movies_df['to_cossim'] = movies_df[['Year', 'quarter']].apply(lambda x: ''.join(x), axis=1)
        # Define a TF-IDF Vectorizer Object. Remove all english stopwords
        tfidf = TfidfVectorizer(stop_words='english')

        # Replace NaN with an empty string
        df_cossim.fillna("", inplace=True)
        tfidf_matrix = tfidf.fit_transform(df_cossim[:n])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        #print(tfidf_matrix.shape, cosine_sim.shape)
        #print(cosine_sim.shape)
        self.set_model_key(feature_list)
        pd.DataFrame(data=cosine_sim, index=movies_df.id[:n], columns=movies_df.id[:n]).to_csv(
            "C:/datasets/the-movies-dataset/models/content_based/content_" + self.model_key + "_"+self.model_type+".csv")

    def make_recommendation(self, title, model_key, n):
        # Obtain the id of the movie that matches the title
        Cnt = Content()
        Cnt.load_content_list()
        Cnt.movielist.set_index("title", inplace=True)
        idx = Cnt.get_id_by_title(title)
        print("Finding similar movies based on ", idx, title, "using", self.model_type, model_key)
        sim_matrix = self.get_model(model_key, self.model_type)
        movie_sim = sim_matrix[["id", str(idx)]].rename(columns={str(idx): 'similarity'}).set_index("id")
        movie_sim = movie_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
        Cnt.movielist.reset_index().set_index("id", inplace=True)
        return pd.concat([Cnt.get_contents_by_id_list(movie_sim.index.values.tolist()), movie_sim], axis=1)

class KNN_Recommender(Base_Recommender):

    def __init__(self):
        Base_Recommender.__init__(self)
        self.model_type = "knn"

    def create_model(self, feature_list):
        n = 10000
        movies_df = self.get_data()
        df_cossim = movies_df[feature_list].fillna("").apply(lambda x: ''.join(x), axis=1)
        #print(movies_df)
        #movies_df['to_cossim'] = movies_df[['Year', 'quarter']].apply(lambda x: ''.join(x), axis=1)
        # Define a TF-IDF Vectorizer Object. Remove all english stopwords
        tfidf = TfidfVectorizer(stop_words='english')


        # Replace NaN with an empty string
        df_cossim.fillna("", inplace=True)
        tfidf_matrix = tfidf.fit_transform(df_cossim)
        #print(tfidf_matrix)
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20).fit(tfidf_matrix[:n]) #metric cosine
        distances, indices = knn.kneighbors(tfidf_matrix[:n])
        #print(indices.shape)
        #print(distances.shape)
        nn_movie_ids = np.empty(indices.shape)
        #print(nn_movie_ids.shape, "empty")
        i = 0
        for line in indices:
            #ele_ids = np.array([movies_df.loc[ele, "id"] for ele in line]).reshape(1, nn_movie_ids.shape[1])
            nn_movie_ids[i] = [movies_df.loc[ele, "id"] for ele in line]
            i += 1
        #print(nn_movie_ids.shape)
        #print(nn_movie_ids)
        #print(movies_df["id"].values.shape, nn_movie_ids.shape, distances.shape)
        data = np.concatenate((movies_df["id"][:n].values.reshape(-1,1), nn_movie_ids, distances), axis=1)
        self.set_model_key(feature_list)
        pd.DataFrame(data=data, columns=["id"]+["sid_"+str(i) for i in range(20)]+["dist_"+str(i) for i in range(20)]).to_csv(
            "C:/datasets/the-movies-dataset/models/content_based/content_" + self.model_key + "_"+self.model_type+".csv")


    def make_recommendation(self, title, model_key, n):
        # Obtain the id of the movie that matches the title
        Cnt = Content()
        Cnt.load_content_list()
        Cnt.movielist.set_index("title", inplace=True)
        idx = Cnt.get_id_by_title(title)
        print("Finding similar movies based on ", idx, title, "using", self.model_type, model_key)
        knn_dist = self.get_model(model_key, self.model_type).set_index("id")
        #print(knn_dist)
        movie_sim = knn_dist.loc[idx, ["sid_"+str(i) for i in range(1,n+1)]].values.ravel().astype("int")
        movie_dist = knn_dist.loc[idx, ["dist_" + str(i) for i in range(1,n+1)]].values.reshape(-1, 1)
        Cnt.movielist.reset_index().set_index("id", inplace=True)
        # print(movie_sim.align(Cnt.movielist, join="left", axis=0))
        #print(movie_sim.tolist())
        df = Cnt.get_contents_by_id_list(movie_sim.tolist())
        df["similarity"] = movie_dist
        #print(df)
        return df

class TSVD_Recommender(Base_Recommender):

    def __init__(self):
        Base_Recommender.__init__(self)
        self.model_type = "tsvd"

    def create_model(self, feature_list):
        n = 10000
        movies_df = self.get_data()
        #print(movies_df)
        df_cossim = movies_df[feature_list].fillna("").apply(lambda x: ''.join(x), axis=1)
        #print(movies_df)
        #movies_df['to_cossim'] = movies_df[['Year', 'quarter']].apply(lambda x: ''.join(x), axis=1)
        # Define a TF-IDF Vectorizer Object. Remove all english stopwords
        tfidf = TfidfVectorizer(stop_words='english')

        # Replace NaN with an empty string
        df_cossim.fillna("", inplace=True)
        tfidf_matrix = tfidf.fit_transform(df_cossim)
        #print(tfidf_matrix)
        svd = TruncatedSVD(n_components=12, random_state=17)
        svd_matrix = svd.fit_transform(tfidf_matrix[:n])
        #print(svd_matrix.shape)
        #print(svd_matrix)
        movie_sim = np.corrcoef(svd_matrix)
        #print(movie_sim.shape)
        self.set_model_key(feature_list)
        pd.DataFrame(data=movie_sim, index=movies_df.id[:n], columns=movies_df.id[:n]).to_csv(
            "C:/datasets/the-movies-dataset/models/content_based/content_" + self.model_key + "_"+self.model_type+".csv")

    def make_recommendation(self, title, model_key, n):
        # Obtain the id of the movie that matches the title
        Cnt = Content()
        Cnt.load_content_list()
        Cnt.movielist.set_index("title", inplace=True)
        idx = Cnt.get_id_by_title(title)
        print("Finding similar movies based on ", idx, title, "using", self.model_type, model_key)
        sim_matrix = self.get_model(model_key, self.model_type)
        movie_sim = sim_matrix[["id",str(idx)]].rename(columns={str(idx): 'similarity'}).set_index("id")
        movie_sim = movie_sim.sort_values(by=["similarity"],ascending=False)[1:n + 1]
        Cnt.movielist.reset_index().set_index("id", inplace=True)
        return pd.concat([Cnt.get_contents_by_id_list(movie_sim.index.values.tolist()), movie_sim], axis=1)

if __name__ == "__main__":

    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    CS_Rec = CosSim_Recommender()
    #CS_Rec.create_model(["overview"])
    print(CS_Rec.make_recommendation("The Toy", ["leads", "genres"], 10))
    #print(text_cos_sim_recommender("The Toy", "overview")[["imdb_id", "title", "overview", "tagline", "genres"]])

    KNN = KNN_Recommender()
    #KNN.create_model(["leads", "genres"])
    #print(KNN.make_recommendation("The Toy", ["leads", "genres"], 10))

    SVD = TSVD_Recommender()
    #SVD.create_model(["leads", "genres"])
    print(SVD.make_recommendation("The Toy", ["leads", "genres"], 10))
