import pandas as pd
from reco_engine.prepare_data import read_movie_ratings
import sys
from sklearn.metrics.pairwise import cosine_similarity
from reco_engine.Lib_Content import Content, Content_Helper
from reco_engine.Lib_User import User, User_Helper
import numpy as np
from surprise import SVD, SVDpp, Dataset, Reader, evaluate
from surprise.model_selection import cross_validate
from sklearn.externals import joblib
from scipy import sparse


class Base_Recommender():

    def __init__(self, type="movie"):
        self.model_key = type
        if type == "movie":
            self.pv_index = "id"
            self.pv_columns = "userId"
        else:
            self.pv_index = "userId"
            self.pv_columns = "id"
        self.matrix_filename = "C:/datasets/the-movies-dataset/models/collaborative_based/coll_" + type + ".npz"
        self.matrix_keys_filename = "C:/datasets/the-movies-dataset/models/collaborative_based/coll_" + type + "_keys.csv"
    def get_ratings(self):
        ratings = pd.read_csv(r'C:\datasets\the-movies-dataset\prep_ratings.csv')
        return ratings

    def get_model(self):
        matrix = sparse.load_npz(self.matrix_filename)
        keys = np.loadtxt(self.matrix_keys_filename)
        keys = [str(int(x)) for x in keys]
        return pd.DataFrame(data=matrix.todense(), columns=keys, index=keys)


class CosSim_Recommender(Base_Recommender):
    def __init__(self, type):
        Base_Recommender.__init__(self, type)

    def create_model(self):
        n=61000
        raw_data = self.get_ratings()[:n].fillna(0)
        raw_matrix=raw_data.pivot_table(values='rating', index=self.pv_index, columns=self.pv_columns).fillna(0)
        #print(coll_matrix)
        #print(coll_matrix.mean(axis=1))
        if self.model_key == "user":
            coll_matrix = raw_matrix.values-raw_matrix.mean(axis=1).values.reshape(-1,1)
            coll_matrix = pd.DataFrame(data = coll_matrix, index=raw_matrix.index, columns =raw_matrix.columns )
        else:
            coll_matrix = raw_matrix
        cosine_sim = cosine_similarity(coll_matrix, coll_matrix)
        np.savetxt(self.matrix_keys_filename, raw_data[self.pv_index ].unique(), fmt="%s", delimiter=",")
        sparse.save_npz(self.matrix_filename, sparse.csr_matrix(cosine_sim))

        #     "C:/datasets/the-movies-dataset/models/collaborative_based/coll_" + self.model_key + ".csv")
        # pd.DataFrame(data=cosine_sim, index=ele_lst, columns=ele_lst).to_csv(
        #     "C:/datasets/the-movies-dataset/models/collaborative_based/coll_" + self.model_key + ".csv")

    def make_recommendation_by_movie(self, idx, n):
        # Obtain the id of the movie that matches the title
        Cnt = Content(idx)
        print("Finding similar movies based on ", idx, Cnt.content["title"], "using", self.model_key)
        model = self.get_model()
        if str(idx) in model.columns.values.tolist():
            movie_sim = model[[str(idx)]].rename(columns={str(idx): 'similarity'})
            movie_sim = movie_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
            print(movie_sim)
            sim_keys = [int(k) for k in movie_sim.index.values.tolist()]
            print(sim_keys)
            print(Content_Helper.get_contents_by_id_list(sim_keys))
            return pd.concat([Content_Helper.get_contents_by_id_list(sim_keys)[["title"]], movie_sim["similarity"]], axis=1)[:n]
        else:
            return pd.DataFrame(columns=["title", "similarity"])

    def make_recommendation_by_user(self, idx, n):
        # Obtain the id of the movie that matches the title
        Usr = User(idx)
        #print(Usr.user)
        print("Finding similar movies for ", idx, Usr.user["username"], "using", self.model_key)
        model = self.get_model()
        #print(model)
        if str(idx) not in model.columns.values.tolist():
            return pd.DataFrame(columns=["title", "pre_rating"])
        user_sim = model[[str(idx)]].rename(columns={str(idx): 'similarity'})
        user_sim = user_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
        #print(user_sim)
        user_sim["similarity"] = user_sim["similarity"]/user_sim["similarity"].sum()
        #print(user_sim)
        #print("similar_users")
        users = [int(id) for id in user_sim.index.values.tolist()]
        #print(pd.concat([Usr.get_users_by_id_list(user_sim.index.values.tolist()), user_sim], axis=1))
        ratings = self.get_ratings().pivot_table(values="rating", columns=self.pv_index, index=self.pv_columns).fillna(0)[users]
        #print(user_sim.index.values.tolist())
        #print(ratings.loc[862,user_sim.index.values.tolist()])
        #print(ratings)
        predicted_ratings = pd.DataFrame(data=np.dot(ratings, user_sim), index = ratings.index.values.tolist(), columns=["pre_rating"])
        #print(predicted_ratings)
        predicted_ratings.index.name = "id"
        predicted_ratings.drop(labels=Usr.get_rating_history().index,inplace=True, axis=0)
        #print(predicted_ratings)
        #print(Cnt.movielist.set_index("id"))
        predicted_ratings= predicted_ratings.join(Content_Helper.get_content_list()["title"], how="left")
        #print(predicted_ratings.sort_values(by=["pre_rating"], ascending=False))
        #print(self.get_data()[self.get_data()["userId"]==11].sort_values(by=["rating"], ascending=False))
        #print(predicted_ratings)
        return predicted_ratings.sort_values(by=["pre_rating"], ascending=False)[:n].reset_index()

class SVD_Recommender(Base_Recommender):
    def __init__(self, type):
        Base_Recommender.__init__(self, type)

    def create_model(self):
        n = 1000

        raw_data = self.get_data()[:n].fillna(0)[["userId", "id", "rating"]]
        reader = Reader()
        data = Dataset.load_from_df(raw_data, reader)
        data.split(n_folds=5)
        svdpp = SVDpp()
        trainset = data.build_full_trainset()
        svdpp.fit(trainset)
        l_usr = User_Helper.get_user_list()
        l_cnt = Content_Helper.get_content_list()
        filename = "C:/datasets/the-movies-dataset/models/collaborative_based/coll_svdpp.sav"
        #col_sid = ["sid_" + str(i) for i in range(1, n + 1)]
        #col_sim = ["sim_" + str(i) for i in range(1, n + 1)]
        # predicted_ratings = pd.DataFrame(index=l_usr["userid"].values.tolist(), columns=l_cnt.index.tolist())
        # #print(predicted_ratings)
        # #print(l_cnt)
        # i = 0
        # for c in l_cnt.index.tolist():
        #     #print(c.id)
        #     for u in l_usr["userid"].values.tolist():
        #         print(i, len(l_cnt)* len(l_usr))
        #         predicted_ratings.loc[u, c] = svdpp.predict(u, c).est
        #         #predicted_ratings.loc[u["userId"],id] = svdpp.predict(u["userId"], c["id"]).est
        # #movies['est'] = raw_data['id'].apply(lambda x: svd.predict(userId, id_to_title.loc[x]['movieId']).est)
        #         i += 1
        # print(predicted_ratings)
        joblib.dump(svdpp, filename)

    def make_recommendation_by_user_movie(self, userid, id):
        svdpp = joblib.load("C:/datasets/the-movies-dataset/models/collaborative_based/coll_svdpp.sav")
        return svdpp.predict(userid, id).est

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #SVD_Rec = SVD_Recommender("user")
    #SVD_Rec.create_model()
    #from surprise.model_selection import train_test_split


    # Load the movielens-100k dataset (download it if needed),
    #data = Dataset.load_builtin('ml-100k')
    #trainset, testset = train_test_split(data, test_size=.25)
    #print(trainset)
    #ratings = read_movie_ratings()
    #CS_Rec = CosSim_Recommender("movie")
    #CS_Rec.create_model()
    CS_Rec = CosSim_Recommender("user")
    CS_Rec.create_model()
    CS_Rec.get_model()
    #print(CS_Rec.get_model().shape)
    #print(CS_Rec.get_model())
    #print(CS_Rec.make_recommendation_by_user(22, 10))

