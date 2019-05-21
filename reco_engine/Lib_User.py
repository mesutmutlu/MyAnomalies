import pandas as pd
import numpy as np
import names
from reco_engine.Lib_Content import Content_Helper, Content
from scipy import  sparse

class User:
    def __init__(self, id):
        self.user = User_Helper.get_user_by_id(id)
        self.id = id
    def get_rating_history(self):
        lst_ratings = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_ratings.csv").set_index("userId")
        lst_ratings = lst_ratings.loc[self.id,:]
        contents = Content_Helper.get_contents_by_id_list(lst_ratings.id.unique())
        content_ratings= contents.merge(lst_ratings, left_on="id", right_on="id", how="inner")
        return content_ratings.set_index("id")

    def get_similar_users(self, n):
        user_matrix = sparse.load_npz(r"C:\datasets\the-movies-dataset\models\collaborative_based\coll_user.npz")
        user_matrix_keys = np.loadtxt(r"C:\datasets\the-movies-dataset\models\collaborative_based\coll_user_keys.csv")
        keys = [str(int(x)) for x in user_matrix_keys]
        sim_users = pd.DataFrame(data=user_matrix.todense(), columns=keys, index=keys)[[str(self.id)]]\
            .sort_values(by=[str(self.id)],ascending=False)[1:n+1]
        sim_users.index.name = "userid"
        sim_users = sim_users.rename(columns={str(self.id): 'similarity'}).reset_index()
        lst_sim_users_id = [int(x) for x in sim_users["userid"].values.tolist()]
        users = User_Helper.get_users_by_id_list(lst_sim_users_id)
        return pd.concat([users.reset_index(), sim_users["similarity"]], axis=1)


class User_Helper:

    def __init__(self):
        pass

    @staticmethod
    def generate_user_list():
        ratings = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_ratings.csv")
        lst_userid = ratings["userId"].unique().reshape(-1, 1)
        #print(lst_userid.shape)
        lst_usernames = np.array([names.get_full_name() for i in range(len(lst_userid))]).reshape(-1, 1)
        #print(lst_usernames.shape)
        users = pd.DataFrame(data=np.concatenate((lst_userid, lst_usernames), axis=1), columns=["userid", "username"])
        users.to_csv(r"C:\datasets\the-movies-dataset\users.csv", index=False)

    @staticmethod
    def get_user_list():
        users = pd.read_csv(r"C:\datasets\the-movies-dataset\users.csv")
        return users

    @staticmethod
    def get_user_by_id(id):
        users = User_Helper.get_user_list()
        user_metadata = pd.read_csv(r"C:\datasets\the-movies-dataset\u.user", delimiter="|", header=None)
        user_metadata.columns = ["userid", "age", "sex", "occupation" ,"todrop"]
        user_metadata.drop("todrop", axis=1, inplace=True)
        user_metadata.set_index("userid")
        #print(user_metadata)
        users = users.merge(user_metadata, left_on=["userid"], right_on=["userid"], how="inner" )
        return users.set_index("userid").reindex([id])

    @staticmethod
    def get_users_by_id_list(lst_id):
        users = pd.read_csv(r"C:\datasets\the-movies-dataset\users.csv", index_col="userid")
        return users.reindex(lst_id)



if __name__ == "__main__":

    user = User(11)
    #print(user.user)
    #print(User_Helper.get_users_by_id_list([101, 34]))
    print(User_Helper.get_user_by_id(1))
    (print(User_Helper.get_user_list()))



    #print(Usr.get_username_by_userid(2))
