import pandas as pd
import numpy as np
import names

class User:

    def __init__(self):
        self.users = pd.DataFrame()

    def generate_user_list(self):
        ratings = pd.read_csv(r"C:\datasets\the-movies-dataset\ratings_small.csv")
        lst_userid = ratings["userId"].unique().reshape(-1, 1)
        print(lst_userid.shape)
        lst_usernames = np.array([names.get_full_name() for i in range(len(lst_userid))]).reshape(-1, 1)
        print(lst_usernames.shape)
        users = pd.DataFrame(data=np.concatenate((lst_userid, lst_usernames), axis=1), columns=["userid", "username"])
        users.to_csv(r"C:\datasets\the-movies-dataset\users.csv", index=False)
        #movies.drop("movieId", axis=1, inplace=True)
        #movies.sort_values(by="id", inplace=True)
        #self.movielist = movies

    def load_user_list(self):
        users = pd.read_csv(r"C:\datasets\the-movies-dataset\users.csv")
        #movies.sort_values(by="id", inplace=True)
        self.users = users

    def get_userid_by_name(self, user_name):
        self.load_user_list()
        return self.users[self.users["username"]==user_name]["userid"].values[0]

    def get_username_by_userid(self, user_id):
        self.load_user_list()
        return self.users[self.users["userid"]==user_id]["username"].values[0]


if __name__ == "__main__":

    Usr = User()
    #Usr.generate_user_list()
    print(Usr.get_userid_by_name("David Beck"))
    print(Usr.get_username_by_userid(2))
