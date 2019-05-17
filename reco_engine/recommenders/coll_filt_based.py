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
        model = model.drop(str(idx))
        if str(idx) in model.columns.values.tolist():
            movie_sim = model[[str(idx)]].rename(columns={str(idx): 'similarity'})
            movie_sim = movie_sim.sort_values(by=["similarity"], ascending=False)[1:n + 1]
            #print(movie_sim)
            sim_keys = [int(k) for k in movie_sim.index.values.tolist()]
            return pd.concat([Content_Helper.get_contents_by_id_list(sim_keys).reset_index()[["id","title"]], movie_sim.reset_index()["similarity"]], axis=1)[:n]
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
        #weigted avearge regarding to similarity ventilation over similar top n users
        #user_sim["similarity"] = user_sim["similarity"]/user_sim["similarity"].sum()
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

class SVDPP_Recommender(Base_Recommender):
    def __init__(self, type):
        Base_Recommender.__init__(self, type)

    def create_model(self):
        n = 1000000
        raw_data = self.get_ratings()[:n].fillna(0)[["userId", "id", "rating"]]
        reader = Reader()
        data = Dataset.load_from_df(raw_data, reader)
        data.split(n_folds=5)
        svdpp = SVDpp()
        trainset = data.build_full_trainset()
        svdpp.fit(trainset)
        filename = "C:/datasets/the-movies-dataset/models/collaborative_based/coll_svdpp.sav"
        joblib.dump(svdpp, filename)

    def predict_rating_by_user_movie(self, userid, id):
        svdpp = joblib.load("C:/datasets/the-movies-dataset/models/collaborative_based/coll_svdpp.sav")
        raw_data = self.get_ratings().fillna(0)[["userId", "id", "rating"]]
        raw_data = raw_data.pivot_table(values="rating", index="userId", columns="id")
        mean = raw_data.mean()
        iu = raw_data.groupby('userId').agg("count")
        #print(iu)
        #print(svdpp.yj)
        sum_yj = np.sum(svdpp.yj, axis=1)
        sum_yj = np.dot(iu, sum_yj)
        iu_p = iu.apply(lambda x: np.power(np.linalg.norm(x),-0.5), axis=1)
        #print(sum_yj.shape, iu_p.shape)
        #print((svdpp.pu + (iu_p*sum_yj).values.reshape(-1,1)).shape, np.transpose(svdpp.qi).shape)
        result = np.dot((svdpp.pu + (iu_p*sum_yj).values.reshape(-1,1)), np.transpose(svdpp.qi))
        print(result.shape)
        result = np.add(result, np.array([5]))
        print(svdpp.bi)
        result = result + svdpp.bu.reshape(-1,1)
        result = result + svdpp.bi
        print(result.shape)
        print("----------")
        print(np.interp(result, (result.min(), result.max()), (0.5, 5)))
        # print(raw_data.shape)
        # print(svdpp.bu.shape)
        # print(svdpp.bi.shape)
        #print(np.transpose(svdpp.qi).shape)
        #print(svdpp.pu.shape)
        #print(iu_dot_yj.shape)
        #pr = np.dot(np.transpose(svdpp.qi), (svdpp.pu + ) + (mean + svdpp.bu + svdpp.bi)





        return svdpp.predict(userid, id).est

class SVD_Recommender(Base_Recommender):
    def __init__(self, type):
        Base_Recommender.__init__(self, type)

    def create_model(self):
        n = 1000000
        raw_data = self.get_ratings()[:n].fillna(0)[["userId", "id", "rating"]]
        reader = Reader()
        data = Dataset.load_from_df(raw_data, reader)
        data.split(n_folds=5)
        svd = SVD()
        trainset = data.build_full_trainset()
        svd.fit(trainset)
        filename = "C:/datasets/the-movies-dataset/models/collaborative_based/coll_svd.sav"
        joblib.dump(svd, filename)

    def predict_rating_by_user_movie(self, userid, id):
        svd = joblib.load("C:/datasets/the-movies-dataset/models/collaborative_based/coll_svd.sav")
        print(svd.bu)
        print(svd.bi)
        print(np.transpose(svd.qi).shape)
        print(svd.pu.shape)

        filename = "C:/datasets/the-movies-dataset/models/collaborative_based/coll_svd.pred"
        joblib.dump(np.dot(svd.pu,np.transpose(svd.qi)), filename)
        print(joblib.load(filename)+svd.bi + svd.bu.reshape(-1,1))
        return svd.predict(userid, id).est

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    SVD_Rec = SVDPP_Recommender("user")
    #SVD_Rec.create_model()
    #SVD_Rec.predict_rating_by_user_movie(1, 197)
    #from surprise.model_selection import train_test_split

    #sys.exit()
    # Load the movielens-100k dataset (download it if needed),
    #data = Dataset.load_builtin('ml-100k')
    #trainset, testset = train_test_split(data, test_size=.25)
    #print(trainset)
    #ratings = read_movie_ratings()
    #CS_Rec = CosSim_Recommender("movie")
    #CS_Rec.create_model()
    CS_Rec = CosSim_Recommender("user")
    CS_Rec.create_model()
    #CS_Rec.get_model()
    #print(CS_Rec.get_model().shape)
    #print(CS_Rec.get_model())
    #print(CS_Rec.make_recommendation_by_user(22, 10))

