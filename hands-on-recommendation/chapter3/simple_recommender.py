import pandas as pd
import sys

def weighted_rating(x, m=50, C=5):
    v = x['vote_count']
    R = x['vote_average']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df = pd.read_csv(r"C:\datasets\hands-on-recommendation\movies_metadata.csv")

    print(df.head())

    m = df['vote_count'].quantile(0.90)
    print("vote limit:", m)

    q_movies = df[(df['runtime'] >= 45) & (df['runtime'] <= 300)]
    q_movies = q_movies[q_movies['vote_count'] >= m]
    print(q_movies.shape)

    C = df['vote_average'].mean()
    print("vote averege mean",C)

    q_movies['score'] = q_movies.apply(lambda x: weighted_rating(x, m, C), axis=1)
    print(q_movies.sort_values(by=['score'], ascending=False))