import pandas as pd
import numpy as np
from reco_engine.Contents import Content


def overview_cos_sim_recommender(title):
    # Obtain the id of the movie that matches the title
    Cnt = Content()
    Cnt.load_content_list()
    Cnt.movielist.set_index("title", inplace=True)
    #print(Cnt.movielist)
    idx = Cnt.movielist[Cnt.movielist.index == title]["id"][0]
    print("Finding similar movies based on ", idx, title)

    cosine_sim = pd.read_csv(r"C:\datasets\the-movies-dataset\models\content_based\content_overview_cos_sim.csv").set_index("id")
    cosine_sim = cosine_sim.iloc[idx]
    print(cosine_sim)

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim))
    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    print(sim_scores)
    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]
    print(sim_scores)
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    print(Cnt.movielist.iloc[movie_indices])
# Function to convert all non-integer IDs to NaN

if __name__ == "__main__":
    overview_cos_sim_recommender("Not on the Lips")
