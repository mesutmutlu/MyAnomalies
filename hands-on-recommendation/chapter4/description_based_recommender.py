import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Function that takes in movie title as input and gives recommendations
def content_recommender(title, cosine_sim, df, indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
# Function to convert all non-integer IDs to NaN
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan

# Extract the director's name. If director is not listed, return NaN
def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def generate_list(x):
    if isinstance(x, list):
        names = [ele['name'] for ele in x]
        #Check if more than 3 elements exist. If yes, return only first three.
        #If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []

# Function to sanitize data to prevent ambiguity.
# Removes spaces and converts to lowercase
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

#Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df = pd.read_csv("C:\datasets\hands-on-recommendation\metadata_clean.csv")

    # Import the original file
    orig_df = pd.read_csv('C:\datasets\hands-on-recommendation\movies_metadata.csv', low_memory=False)

    # Add the useful features into the cleaned dataframe
    df['overview'], df['id'] = orig_df['overview'], orig_df['id']

    df.head()
    # Print the head of the cleaned DataFrame
    print(df.head())

    # Define a TF-IDF Vectorizer Object. Remove all english stopwords
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    df['overview'] = df['overview'].fillna('')

    # Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    # Output the shape of tfidf_matrix
    print(tfidf_matrix.shape)

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix[:10000], tfidf_matrix[:10000])

    # Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    print(content_recommender('The Lion King', cosine_sim, df, indices))

    # Load the keywords and credits files
    cred_df = pd.read_csv('C:\datasets\hands-on-recommendation\credits.csv')
    key_df = pd.read_csv('C:\datasets\hands-on-recommendation\keywords.csv')

    # Print the head of the credit dataframe
    print(cred_df.head())

    # Print the head of the keywords dataframe
    print(key_df.head())

    # Clean the ids of df
    df['id'] = df['id'].apply(clean_ids)

    # Filter all rows that have a null ID
    df = df[df['id'].notnull()]

    # Convert IDs into integer
    df['id'] = df['id'].astype('int')
    key_df['id'] = key_df['id'].astype('int')
    cred_df['id'] = cred_df['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    df = df.merge(cred_df, on='id')
    df = df.merge(key_df, on='id')

    # Display the head of the merged df
    print(df.head())

    # Convert the stringified objects into the native python objects
    from ast import literal_eval

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)

    # Print the first cast member of the first movie in df
    print(df.iloc[0]['crew'][0])

    # Define the new director feature
    df['director'] = df['crew'].apply(get_director)

    # Print the directors of the first five movies
    print(df['director'].head())

    print(df.head())

    # Apply the generate_list function to cast and keywords
    df['cast'] = df['cast'].apply(generate_list)
    df['keywords'] = df['keywords'].apply(generate_list)

    # Only consider a maximum of 3 genres
    df['genres'] = df['genres'].apply(lambda x: x[:3])

    # Print the new features of the first 5 movies along with title
    print(df[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

    # Apply the generate_list function to cast, keywords, director and genres
    for feature in ['cast', 'director', 'genres', 'keywords']:
        df[feature] = df[feature].apply(sanitize)

    # Create the new soup feature
    df['soup'] = df.apply(create_soup, axis=1)

    # Display the soup of the first movie
    print(df.iloc[0]['soup'])

    # Define a new CountVectorizer object and create vectors for the soup.
    # Instead of using TF-IDFVectorizer, we will be using CountVectorizer.
    # This is because using TF-IDFVectorizer will accord less weight to actors and directors who have acted and directed in a relatively larger number of movies.

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])

    # Compute the cosine similarity score (equivalent to dot product for tf-idf vectors)
    cosine_sim2 = cosine_similarity(count_matrix[:10000], count_matrix[:10000])
    # Reset index of your df and construct reverse mapping again
    # Reset index of your df and construct reverse mapping again
    df = df.reset_index()
    indices2 = pd.Series(df.index, index=df['title'])

    print(content_recommender('The Lion King', cosine_sim2, df, indices2))