import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from reco_engine.prepare_data import prepare_movies_data
from ast import literal_eval
import datetime

text_cos_sims = ["overview", "cast", "keywords", "leads", "genres", "belongs_to_collection"]
def text_based_cos_sim(df, n):
    # Define a TF-IDF Vectorizer Object. Remove all english stopwords
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    df.fillna('', inplace=True)
    tfidf_matrix = tfidf.fit_transform(df)
    cosine_sim = linear_kernel(tfidf_matrix[:n], tfidf_matrix[:n])
    print(tfidf_matrix.shape, cosine_sim.shape)
    return cosine_sim

def cre_text_cos_sim_model(movies_df, feature):
    n = 10000
    movies_df = pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")
    cos_sim = text_based_cos_sim(movies_df[feature], n)
    print(cos_sim.shape)
    pd.DataFrame(data=cos_sim, index = movies_df.id[:n], columns=movies_df.id[:n]).to_csv("C:/datasets/the-movies-dataset/models/content_based/content_"+feature+"_cos_sim.csv")

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #movies_df = prepare_movies_data()
    movies_df = pd.read_csv("C:/datasets/the-movies-dataset/prep_data.csv")
    print(movies_df.head())

    for f in text_cos_sims:
        print(f, datetime.datetime.now())
        cre_text_cos_sim_model(movies_df, f)