import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from reco_engine.prepare_data import prepare_movies_data

def text_based_cos_sim(df):
    # Define a TF-IDF Vectorizer Object. Remove all english stopwords
    tfidf = TfidfVectorizer(stop_words='english')
    n=20000
    # Replace NaN with an empty string
    df.fillna('', inplace=True)
    tfidf_matrix = tfidf.fit_transform(df)
    cosine_sim = linear_kernel(tfidf_matrix[:n], tfidf_matrix[:n])
    print(tfidf_matrix.shape, cosine_sim.shape)
    return cosine_sim

def crete_text_based_cos_sim_models():
    movies_df = pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")

    movies_df_overview_sim = text_based_cos_sim(movies_df["overview"])
    movies_df["str_cast"] =  movies_df["cast"].apply(' '.join)
    print(movies_df["str_cast"])
    movies_df_overview_sim = text_based_cos_sim(movies_df["overview"].apply())



if __name__ == "__main__":
    #movies_df = prepare_movies_data()
    movies_df = pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")
    print(movies_df.head())
    movies_df["str_cast"] = movies_df["cast"].apply(lambda x: ' '.join(x))
    print(movies_df["str_cast"])
    #print(text_based_cos_similarity(movies_df["overview"]))