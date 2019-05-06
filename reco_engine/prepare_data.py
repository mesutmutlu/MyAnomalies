import pandas as pd
import numpy as np
import sys
from ast import literal_eval
import datetime

df_movies_fields_to_list = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
df_supp_fields_to_list = ['cast', 'keywords']
keys = dict()
keys["genres"] = "name"
keys["production_companies"] = "name"
keys["production_countries"] = "iso_3166_1"
keys["spoken_languages"] = "iso_639_1"
keys["cast"] = "name"
keys["keywords"] = "name"
keys["belongs_to_collection"] = "name"

def read_movie_metadata_files():
    movies_df = pd.read_csv(r'C:\datasets\the-movies-dataset\movies_metadata.csv', low_memory=False)
    cred_df = pd.read_csv(r'C:\datasets\the-movies-dataset\credits.csv')
    key_df = pd.read_csv(r'C:\datasets\the-movies-dataset\keywords.csv')

    return movies_df, cred_df, key_df

def get_director(x):
    #print(x)
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return [crew_member['name']]
    return []

def get_list(x, c, n):
    if isinstance(x, list):
        names = [ele[keys[c]] for ele in x]
        #Check if more than 3 elements exist. If yes, return only first three.
        #If no, return entire list.
        if len(names) > n:
            names = names[:n]
        return names
    # Return empty list in case of missing/malformed data
    return []

def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan

def sanitize(x):
    if isinstance(x, list):
        #Strip spaces and convert to lowercase
        lst = [str.lower(i.replace(" ", "")) for i in x]
        return ' '.join(lst)
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def get_leads(x, n):
    return x[:n]

def get_collection(x):

    if x != np.NaN:
        try:
            return [literal_eval(x)["name"]]
        except:
            return []
    else:
        return []

def get_date(x, format, g):
    try:
        o_x = datetime.datetime.strptime(x, format)
        if g == "year":
            return int(o_x.year)
        elif g == "month":
            return int(o_x.month)
        elif g == "day":
            return int(o_x.day)
        else:
            return np.NaN
    except:
        return np.NaN

def prepare_movies_metadata():

    movies_df, cred_df, key_df = read_movie_metadata_files()
    #print(movies_df)
    movies_df['id'] = movies_df['id'].apply(clean_ids)
    movies_df = movies_df[movies_df['id'].notnull()]
    key_df['id'] = key_df['id'].apply(clean_ids)
    key_df = key_df[key_df['id'].notnull()]
    cred_df['id'] = cred_df['id'].apply(clean_ids)
    cred_df = cred_df[cred_df['id'].notnull()]

    movies_df['id'] = movies_df['id'].astype('int')
    key_df['id'] = key_df['id'].astype('int')
    cred_df['id'] = cred_df['id'].astype('int')

    movies_df = movies_df.merge(cred_df, on='id')
    movies_df = movies_df.merge(key_df, on='id')

    nan_columns = movies_df.columns[movies_df.isna().any()]
    #prepare movies_df

    movies_df["release_year"] = movies_df["release_date"].apply(lambda x: get_date(x, "%Y-%m-%d", "year"))
    movies_df["release_month"] = movies_df["release_date"].apply(lambda x: get_date(x, "%Y-%m-%d", "month"))
    movies_df["release_day"] = movies_df["release_date"].apply(lambda x: get_date(x, "%Y-%m-%d", "day"))
    if 1==1:

        movies_df["belongs_to_collection"] = movies_df["belongs_to_collection"].apply(
            lambda x: get_collection(x))
        movies_df["director"] = movies_df["crew"].apply(lambda x: get_director(literal_eval(x)))
        for c in df_movies_fields_to_list + df_supp_fields_to_list:
            if c in nan_columns:
                movies_df[c].fillna("[]", inplace=True)
            #convert strings to objects
            movies_df[c] = movies_df[c].apply(literal_eval)
            #convert objects to string
            movies_df[c] = movies_df[c].apply(lambda x: get_list(x, c, 10))

        movies_df["leads"] = movies_df["cast"].apply(lambda x: get_leads(x, 3))


    for feature in df_movies_fields_to_list + df_supp_fields_to_list + ['leads', 'director', 'belongs_to_collection']:
        movies_df[feature] = movies_df[feature].apply(sanitize)

    drop_cols = ["budget", "original_title", "popularity", "poster_path","revenue", "video", "crew", "homepage"]
    movies_df.drop(drop_cols, inplace=True, axis=1)
    #print(movies_df)
    movieIds = pd.read_csv(r"C:\datasets\the-movies-dataset\movie_ids.csv")
    return pd.merge(movies_df, movieIds[["id", "movieId"]], on=['id', 'id'], how='inner')[movies_df.columns.values.tolist()]

def read_movie_ratings():

    movieIds = pd.read_csv(r"C:\datasets\the-movies-dataset\movie_ids.csv")
    prep_data = pd.read_csv(r"C:\datasets\the-movies-dataset\prep_data.csv")[["id"]]
    df = pd.merge(movieIds, prep_data, on=['id', 'id'], how='inner')
    prep_data = None
    ratings = pd.read_csv(r'C:\datasets\the-movies-dataset\ratings.csv')

    #print(ratings)
    #df = pd.merge(ratings, df, on=['movieId', 'movieId'], how='inner')[ratings.columns.values.tolist()+["id"]]
    df = ratings[ratings.movieId.isin(df.movieId.unique())]
    ratings = None
    print(df)
    users = pd.read_csv(r"C:\datasets\the-movies-dataset\users.csv")
    print(users)
    #df = pd.merge(df, users, on=['userId', 'userid'], how='inner')
    df = df[df.userId.isin(users.userid.unique())]
    users = None
    df = pd.merge(df, movieIds, on=['movieId', 'movieId'], how='inner')[df.columns.values.tolist()+["id"]]
    print(df)
    df["id"] = df["id"].astype(int)
    df["id"] = df["id"].astype(str)
    return df

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #prepare_movies_metadata()
    #print(read_movie_ratings())
    #movies_df = prepare_movies_metadata()

    #prep_ratings = read_movie_ratings()
    prep_ratings = pd.read_csv("C:/datasets/the-movies-dataset/prep_ratings.csv")
    print(prep_ratings)
    prep_ratings["id"] = prep_ratings["id"].astype(int)
    prep_ratings["id"] = prep_ratings["id"].astype(int).astype(str)
    prep_ratings.to_csv("C:/datasets/the-movies-dataset/prep_ratings.csv", index=False)
    #movies_df.to_csv("C:/datasets/the-movies-dataset/prep_movies_mdata.csv", index=False)
    #movies_df = pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")
    #print(movies_df = read_movie_ratings()[movies_df["id"]==23805])
    #23805

    #lang_codes_df = pd.read_csv(r'C:\datasets\the-movies-dataset\language-codes-full.csv')
    #print(lang_codes_df.head())