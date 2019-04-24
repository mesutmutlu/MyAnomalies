import pandas as pd
import sys
import numpy as np
from ast import literal_eval

#Helper function to convert NaT to 0 and all other years to integers.
def convert_int(x):
    try:
        return int(x)
    except:
        return 0

def weighted_rating(x, m=50, C=5):
    v = x['vote_count']
    R = x['vote_average']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)

def build_chart(gen_df, percentile=0.8):
    # Ask for preferred genres
    print("Input preferred genre")
    genre = input()

    # Ask for lower limit of duration
    print("Input shortest duration")
    low_time = int(input())

    # Ask for upper limit of duration
    print("Input longest duration")
    high_time = int(input())

    # Ask for lower limit of timeline
    print("Input earliest year")
    low_year = int(input())

    # Ask for upper limit of timeline
    print("Input latest year")
    high_year = int(input())

    # Define a new movies variable to store the preferred movies. Copy the contents of gen_df to movies
    movies = gen_df.copy()

    # Filter based on the condition
    movies = movies[(movies['genre'] == genre) &
                    (movies['runtime'] >= low_time) &
                    (movies['runtime'] <= high_time) &
                    (movies['year'] >= low_year) &
                    (movies['year'] <= high_year)]

    print(movies.shape)
    # Compute the values of C and m for the filtered movies
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(percentile)
    print(C, m)

    # Only consider movies that have higher than m votes. Save this in a new dataframe q_movies

    q_movies = movies.copy().loc[movies['vote_count'] >= m]
    print(q_movies.shape)

    # Calculate score using the IMDB formula
    q_movies['score'] = q_movies.apply(lambda x:  weighted_rating(x, m, C), axis=1)

    # Sort movies in descending order of their scores
    q_movies = q_movies.sort_values('score', ascending=False)

    return q_movies

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df = pd.read_csv(r"C:\datasets\hands-on-recommendation\movies_metadata.csv")

    print(df.columns)
    df = df[['title', 'genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Convert release_date into pandas datetime format
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Extract year from the datetime
    df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    df['year'] = df['year'].apply(convert_int)

    # Drop the release_date column
    df = df.drop('release_date', axis=1)

    print(df.head())

    print(df.iloc[0]['genres'])

    # Convert all NaN into stringified empty lists
    df['genres'] = df['genres'].fillna('[]')

    # Apply literal_eval to convert to the list object
    df['genres'] = df['genres'].apply(literal_eval)
    print(df.head())

    # Convert list of dictionaries to a list of strings
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    print(df["year"].max())
    print(df.head())

    df.to_csv('C:/datasets/hands-on-recommendation/metadata_clean.csv', index=False)
    # Create a new feature by exploding genres
    s = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    print(s)
    # Name the new feature as 'genre'
    s.name = 'genre'

    # Create a new dataframe gen_df which by dropping the old 'genres' feature and adding the new 'genre'.
    gen_df = df.drop('genres', axis=1).join(s)

    print("Print the head of the new gen_df")
    # Print the head of the new gen_df
    print(gen_df.head())

    print(build_chart(gen_df).head())