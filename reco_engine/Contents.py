import pandas as pd
import sys
class Content:

    def __init__(self):
        self.movielist = pd.DataFrame()
        pass

    def load_content_list(self):
        self.movielist = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_data.csv")
        #movies.sort_values(by="id", inplace=True)

    def get_title_by_id(self, movie_id):
        self.load_content_list()
        print(self.movielist)
        return self.movielist[self.movielist["id"] == movie_id]["title"].values[0]

    def get_id_by_title(self, movie_title):
        self.load_content_list()
        return self.movielist[self.movielist["title"] == movie_title]["id"].values[0]

    def get_contents_by_id_list(self, lst_id):
        self.load_content_list()
        return self.movielist.set_index("id").loc[lst_id,]

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    Cnt = Content()
    print(Cnt.get_contents_by_id_list([23805, 47439, 92331, 507, 30970, 26243, 24086, 6715, 36998, 15514]))
    #23805, 47439, 92331, 507, 30970, 26243, 24086, 6715, 36998, 15514
