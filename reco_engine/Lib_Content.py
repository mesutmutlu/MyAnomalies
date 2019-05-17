import pandas as pd
import sys
class Content:
    def __init__(self, id):
        self.id = id
        #hlp_cnt = Content_Helper()
        self.content = Content_Helper.get_content_by_id(id)

class Content_Helper:
    def __init__(self):
        pass

    @staticmethod
    def get_content_by_id(id):
        lst_cnt = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_data.csv").set_index("id")
        return lst_cnt.reindex([id])

    @staticmethod
    def get_content_list():
        lst_cnt = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_data.csv").set_index("id")
        return lst_cnt

    @staticmethod
    def get_id_by_title(movie_title):
        lst_cnt = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_data.csv")
        return lst_cnt[lst_cnt["title"] == movie_title]["id"].values[0]

    @staticmethod
    def get_contents_by_id_list(lst_id):
        lst_cnt = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_data.csv").set_index("id")
        return lst_cnt.reindex(lst_id)

    @staticmethod
    def get_content_ids_by_movieid(lst_id):
        df_movieId = pd.read_csv(r"C:\datasets\the-movies-dataset\movie_ids.csv")
        i =1
        for x in lst_id:
            if x in df_movieId['movieId']:
                print(i, "existance error", x, df_movieId[df_movieId['movieId']==x]["id"])
            i += 1
        return df_movieId[df_movieId['movieId'].isin(lst_id)][["movieId","id"]]

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(User_Helper.get_user_list())
    #23805, 47439, 92331, 507, 30970, 26243, 24086, 6715, 36998, 15514
