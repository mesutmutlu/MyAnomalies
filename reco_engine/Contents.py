import pandas as pd
class Content:

    def __init__(self):
        pass

    def load_content_list(self):
        movies = pd.read_csv(r"C:\datasets\the-movies-dataset\movie_ids.csv")
        movies.drop("movieId", axis=1, inplace=True)
        #movies.sort_values(by="id", inplace=True)
        self.movielist = movies

    def get_title_by_id(self, movie_id):
        self.load_content_list()
        return self.movielist [self.movielist ["id"]==movie_id]["title"].values[0]

    def get_id_by_title(self, movie_title):
        self.load_content_list()
        return self.movielist [self.movielist ["title"]==movie_title]["id"].values[0]


if __name__ == "__main__":

    Cnt = Content()
    print(Cnt.get_title_by_id(2))
    print(Cnt.get_id_by_title("Ariel"))
