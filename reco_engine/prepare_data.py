import pandas as pd
import numpy as np
import sys
from ast import literal_eval

df_movies_fields_to_list = ['genres', 'production_companies', 'production_countries', 'spoken_languages']
df_supp_fields_to_list = ['cast', 'crew', 'keywords']

def read_files():
    movies_df = pd.read_csv(r'C:\datasets\the-movies-dataset\movies_metadata.csv', low_memory=False)
    cred_df = pd.read_csv(r'C:\datasets\the-movies-dataset\credits.csv')
    key_df = pd.read_csv(r'C:\datasets\the-movies-dataset\keywords.csv')

    return movies_df, cred_df, key_df

def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return [crew_member['name']]
    return []

def get_list(x, c, n):
    keys = dict()
    keys["genres"] = "name"
    keys["production_companies"] = "name"
    keys["production_countries"] = "iso_3166_1"
    keys["spoken_languages"] = "iso_639_1"
    keys["cast"] = "name"
    keys["keywords"] = "name"
    if isinstance(x, list):
        #print(x)
        if c != "crew":
            names = [ele[keys[c]] for ele in x]
            #Check if more than 3 elements exist. If yes, return only first three.
            #If no, return entire list.
            if len(names) > n:
                names = names[:n]
            return names
        else:
            return get_director(x)
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
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def get_leads(x, n):
    return x[:n]


def prepare_movies_data():
    print()
    movies_df, cred_df, key_df = read_files()

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
    if 1==1:

        for c in df_movies_fields_to_list + df_supp_fields_to_list:
            if c in nan_columns:
                movies_df[c].fillna("[]", inplace=True)
            #convert strings to objects
            movies_df[c] = movies_df[c].apply(literal_eval)
            #convert objects to string
            movies_df[c] = movies_df[c].apply(lambda x: get_list(x, c, 10))
        movies_df["leads"] = movies_df["cast"].apply(lambda x: get_leads(x, 3))
    for feature in df_movies_fields_to_list + df_supp_fields_to_list + ['leads']:
        movies_df[feature] = movies_df[feature].apply(sanitize)

    drop_cols = ["budget", "original_title", "popularity", "poster_path","revenue", "video"]
    movies_df.drop(drop_cols, inplace=True, axis=1)
    #print(movies_df[45459:45465])
    return movies_df

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    movies_df = prepare_movies_data()
    movies_df.to_csv("C:/datasets/the-movies-dataset/prep_data.csv", index=False)
    lang_codes_df = pd.read_csv(r'C:\datasets\the-movies-dataset\language-codes-full.csv')
    #print(lang_codes_df.head())