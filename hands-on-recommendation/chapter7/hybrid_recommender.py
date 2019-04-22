import pandas as pd
import numpy as np
#Build the SVD based Collaborative filter
from surprise import SVD, Reader, Dataset


def hybrid(userId, title):
    # Extract the cosine_sim index of the movie
    idx = cosine_sim_map[title]

    # Extract the TMDB ID of the movie
    tmdbId = title_to_id.loc[title]['id']

    # Extract the movie ID internally assigned by the dataset
    movie_id = title_to_id.loc[title]['movieId']

    # Extract the similarity scores and their corresponding index for every movie from the cosine_sim matrix
    sim_scores = list(enumerate(cosine_sim[str(int(idx))]))

    # Sort the (index, score) tuples in decreasing order of similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top 25 tuples, excluding the first
    # (as it is the similarity score of the movie with itself)
    sim_scores = sim_scores[1:26]

    # Store the cosine_sim indices of the top 25 movies in a list
    movie_indices = [i[0] for i in sim_scores]

    # Extract the metadata of the aforementioned movies
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]

    # Compute the predicted ratings using the SVD filter
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, id_to_title.loc[x]['movieId']).est)

    # Sort the movies in decreasing order of predicted rating
    movies = movies.sort_values('est', ascending=False)

    # Return the top 10 movies as recommendations
    return movies.head(10)

if __name__ == "__main__":

    # Import or compute the cosine_sim matrix, this is movie similarity matrix
    cosine_sim = pd.read_csv(r'C:\datasets\the-movies-dataset\cosine_sim.csv')
    print(cosine_sim)

    # Import or compute the cosine sim mapping matrix
    cosine_sim_map = pd.read_csv(r'C:\datasets\the-movies-dataset\cosine_sim_map.csv', header=None)
    print(cosine_sim_map)

    # Convert cosine_sim_map into a Pandas Series
    cosine_sim_map = cosine_sim_map.set_index(0)
    cosine_sim_map = cosine_sim_map[1]
    reader = Reader()
    ratings = pd.read_csv(r'C:\datasets\the-movies-dataset\ratings_small.csv')
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    data.split(n_folds=5)
    svd = SVD()
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    # Build title to ID and ID to title mappings
    id_map = pd.read_csv(r'C:\datasets\the-movies-dataset\movie_ids.csv')
    id_to_title = id_map.set_index('id')
    title_to_id = id_map.set_index('title')

    # Import or compute relevant metadata of the movies
    smd = pd.read_csv(r'C:\datasets\the-movies-dataset\metadata_small.csv')

    print(hybrid(1, 'Avatar'))

    print(hybrid(2, 'Avatar'))