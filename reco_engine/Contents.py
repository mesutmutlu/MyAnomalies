import pandas as pd
class Content:

    def __init__(self):
        self.movielist = pd.DataFrame()
        pass

    def load_content_list(self):
        movies = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_data.csv")
        #movies.sort_values(by="id", inplace=True)
        self.movielist = movies[["id","title"]]

    def get_title_by_id(self, movie_id):
        self.load_content_list()
        print(self.movielist)
        return self.movielist[self.movielist["id"] == movie_id]["title"].values[0]

    def get_id_by_title(self, movie_title):
        self.load_content_list()
        return self.movielist[self.movielist["title"] == movie_title]["id"].values[0]


if __name__ == "__main__":

    Cnt = Content()
    Cnt.load_content_list()
    print(Cnt.movielist)
    print(Cnt.movielist[Cnt.movielist["id"]==51945]["title"].values[0])
