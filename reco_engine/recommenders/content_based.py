import pandas as pd
import numpy as np
from reco_engine.Lib_Content import Content, Content_Helper
import sys
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.neighbors import NearestNeighbors, BallTree
import pickle
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, coo_matrix
from nltk.corpus import stopwords
import nltk
from nltk.stem import  WordNetLemmatizer


class Base_Recommender():

    def __init__(self, features):
        #self.all_features = ["overview", "cast", "keywords", "leads", "genres", "belongs_to_collection"]
        self.set_model_key(features)
        self.model_type = ""

    def get_data(self):
        #self.raw_data = pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")
        return pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")

    def get_model(self):
        #self.set_model_key(feature_list)
        print("model:","C:\datasets\the-movies-dataset\models\content_based\content_" + self.model_key + "_"+self.model_type+".csv")
        return pd.read_csv(
            r"C:\datasets\the-movies-dataset\models\content_based\content_" + self.model_key + "_"+self.model_type+".csv")

    def set_model_key(self, feature_list):
        if len(feature_list) == 1:
            self.model_key = feature_list[0]
        elif type(feature_list) == str:
            self.model_key = feature_list
        else:
            self.model_key = "_".join(feature_list)

class Cosine_Recommender(Base_Recommender):

    def __init__(self, features):
        Base_Recommender.__init__(self, features)
        self.model_type = "cosine"

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        stems = []
        for item in tokens:
            stems.append(WordNetLemmatizer().lemmatize(item))
        return stems

    def create_model(self, feature_list, f_type):
        n = 100000
        num = 20
        movies_df = self.get_data()
        df_cossim = movies_df[feature_list].fillna("").apply(lambda x: ' '.join(x), axis=1)
        #print(movies_df)
        #movies_df['to_cossim'] = movies_df[['Year', 'quarter']].apply(lambda x: ''.join(x), axis=1)
        # Define a TF-IDF Vectorizer Object. Remove all english stopwords
        if f_type == "category":
            tfidf = TfidfVectorizer(analyzer="word")
        elif f_type == "text":
            tfidf = TfidfVectorizer(stop_words=stopwords.words("english"), analyzer="word", tokenizer=self.tokenize)
        else:
            tfidf = TfidfVectorizer(stop_words=stopwords.words("english"), analyzer="word")

        # Replace NaN with an empty string
        df_cossim.fillna("", inplace=True)
        #print(df_cossim)
        tfidf_matrix = tfidf.fit_transform(df_cossim[:n])
        #print(tfidf_matrix)
        #print(tfidf.vocabulary_)
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        #print(tfidf_matrix.shape, cosine_sim.shape)
        #print(cosine_sim)
        self.set_model_key(feature_list)
        df_cosine_sim = pd.DataFrame(data=cosine_sim, index=movies_df.id[:n], columns=movies_df.id[:n])


        data = []
        df_similars2 = pd.DataFrame(columns=["id, similar_id, similarity"])
        for index, line in df_cosine_sim.iterrows():
            #print(index)
            line = line.sort_values(ascending=False)[1:num+1]
            ids = line.index.values.tolist()
            similarity = line.tolist()
            s_line = np.array([index] + ids + similarity).reshape(1,(num*2)+1)
            #print(data.shape, s_line.shape)
            if data != []:
                data = np.concatenate((data,s_line), axis=0)
            else:
                data = s_line
        df_similars = pd.DataFrame(data=data, columns=["id"] + ["sid_" + str(i) for i in range(num)] +
                                                      ["sim_" + str(i) for i in   range(num)])

        for col in ["id"] + ["sid_" + str(i) for i in range(20)]:
            df_similars[col] = df_similars[col].astype(int)

        df_similars.to_csv(
            "C:/datasets/the-movies-dataset/models/content_based/content_" + self.model_key + "_" + self.model_type + ".csv", index=False)

    def make_recommendation(self, idx, n):
        Cnt = Content(idx)
        print("Finding similar movies based on ", idx, Cnt.content["title"], "using", self.model_type, self.model_key)
        sim_movies = self.get_model().set_index("id").reindex([idx])
        movie_sim = sim_movies.loc[idx, ["sid_" + str(i) for i in range(1, n + 1)]].values.ravel().astype("int")
        movie_dist = sim_movies.loc[idx, ["sim_" + str(i) for i in range(1, n + 1)]].values.reshape(-1, 1)
        df = Content_Helper.get_contents_by_id_list(movie_sim.tolist())
        df["similarity"] = movie_dist
        return df[["title","similarity"]]

class KNN_Recommender(Base_Recommender):

    def __init__(self):
        Base_Recommender.__init__(self)
        self.model_type = "knn"

    def create_model(self, feature_list):
        n = 10000
        movies_df = self.get_data()
        df_cossim = movies_df[feature_list].fillna("").apply(lambda x: ' '.join(x), axis=1)
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
            "C:/datasets/the-movies-dataset/models/content_based/content_" + self.model_key + "_"+self.model_type+".csv", index=False)


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
        print(movie_sim.tolist())
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
        df_cossim = movies_df[feature_list].fillna("").apply(lambda x: ' '.join(x), axis=1)
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

    # movies_df = pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")
    # df = pd.DataFrame(columns=["leads", "genres"], data=[["1 2","2"], ["a","b c"]])
    # print(df)
    # df_cossim = df[["leads", "genres"]].fillna("").apply(lambda x: ' '.join(x), axis=1)
    # print(df_cossim)


    #nltk.download('wordnet')
    CS_Rec = Cosine_Recommender(["overview"])
    print(CS_Rec.make_recommendation(862, 10))
    CS_Rec = Cosine_Recommender(["leads", "genres"])
    print(CS_Rec.make_recommendation(862, 10))
    CS_Rec = Cosine_Recommender(["keywords", "genres"])
    print(CS_Rec.make_recommendation(862, 10))
    CS_Rec = Cosine_Recommender(["cast"])
    print(CS_Rec.make_recommendation(862, 10))
    sys.exit()
    # CS_Rec.create_model(["leads", "genres"], "category")
    CS_Rec.create_model(["overview"], "text")
    CS_Rec.create_model(["keywords", "genres"], "category")
    CS_Rec.create_model(["keywords", "genres", "leads"], "category")
    CS_Rec.create_model(["keywords"], "category")
    CS_Rec.create_model(["cast"], "category")
    #print(CS_Rec.make_recommendation("The Toy", ["leads", "genres"], 10))
    #print(text_cos_sim_recommender("The Toy", "overview")[["imdb_id", "title", "overview", "tagline", "genres"]])
    sys.exit()
    KNN = KNN_Recommender()
    #KNN.create_model(["leads", "genres"])
    #print(KNN.make_recommendation("The Toy", ["leads", "genres"], 10))

    SVD = TSVD_Recommender()
    #SVD.create_model(["leads", "genres"])
    print(SVD.make_recommendation("The Toy", ["leads", "genres"], 10))
