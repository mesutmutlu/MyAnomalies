import pandas as pd
import numpy as np
import names
from reco_engine.Lib_Content import Content_Helper, Content

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

class User_Helper:

    def __init__(self):
        pass

    @staticmethod
    def generate_user_list(self):
        ratings = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_ratings.csv")
        lst_userid = ratings["userId"].unique().reshape(-1, 1)
        print(lst_userid.shape)
        lst_usernames = np.array([names.get_full_name() for i in range(len(lst_userid))]).reshape(-1, 1)
        print(lst_usernames.shape)
        users = pd.DataFrame(data=np.concatenate((lst_userid, lst_usernames), axis=1), columns=["userid", "username"])
        users.to_csv(r"C:\datasets\the-movies-dataset\users.csv", index=False)

    @staticmethod
    def get_user_list():
        users = pd.read_csv(r"C:\datasets\the-movies-dataset\users.csv")
        return users

    @staticmethod
    def get_user_by_id(id):
        users = pd.read_csv(r"C:\datasets\the-movies-dataset\users.csv", index_col="userid")
        user_metadata = pd.read_csv(r"C:\datasets\the-movies-dataset\u.user", delimiter="|", header=None)
        user_metadata.columns = ["userid", "age", "sex", "occupation" ,"todrop"]
        user_metadata.drop("todrop", axis=1, inplace=True)
        user_metadata.set_index("userid")
        users = users.merge(user_metadata, left_on=["userid"], right_on=["userid"], how="inner" )
        return users.reindex([id])

    @staticmethod
    def get_users_by_id_list(lst_id):
        users = pd.read_csv(r"C:\datasets\the-movies-dataset\users.csv", index_col="userid")
        return users.reindex(lst_id)





if __name__ == "__main__":

    user = User(2)
    #print(user.user)
    print(user.get_rating_history())


    #print(Usr.get_username_by_userid(2))
