import pandas as pd
import numpy as np
from reco_engine.Contents import Content
import sys


def text_cos_sim_recommender(title, base):
    # Obtain the id of the movie that matches the title
    Cnt = Content()
    Cnt.load_content_list()
    Cnt.movielist.set_index("title", inplace=True)
    idx = Cnt.get_id_by_title(title)
    print("Finding similar movies based on ", idx, title)
    cosine_sim = pd.read_csv(r"C:\datasets\the-movies-dataset\models\content_based\content_"+base+"_cos_sim.csv").set_index("id")
    movie_sim= cosine_sim[str(idx)].sort_values(ascending=False)[1:11]
    Cnt.movielist.reset_index().set_index("id", inplace=True)
    #print(movie_sim.align(Cnt.movielist, join="left", axis=0))
    return pd.concat([Cnt.get_contents_by_id_list(movie_sim.index.values.tolist()), movie_sim], axis=1)

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(text_cos_sim_recommender("The Toy", "overview")[["imdb_id", "title", "overview", "tagline", "genres"]])
