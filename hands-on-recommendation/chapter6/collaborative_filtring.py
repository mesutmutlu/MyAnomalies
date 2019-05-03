import pandas as pd
#Import the train_test_split function
from sklearn.model_selection import train_test_split
#Import the mean_squared_error function
from sklearn.metrics import mean_squared_error
import numpy as np
# Import cosine_score
from sklearn.metrics.pairwise import cosine_similarity
#Import the required classes and methods from the surprise library
from surprise import Reader, Dataset, KNNBasic, evaluate

#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#Define the baseline model to always return 3.
def baseline(user_id, movie_id):
    return 3.0


# Function to compute the RMSE score obtained on the testing set by a model
def score(cf_model):
    # Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])

    # Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie) for (user, movie) in id_pairs])

    # Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])

    # Return the final RMSE score
    return rmse(y_true, y_pred)


# User Based Collaborative Filter using Mean Ratings
def cf_user_mean(user_id, movie_id):
    # Check if movie_id exists in r_matrix
    if movie_id in r_matrix:
        # Compute the mean of all the ratings given to the movie
        mean_rating = r_matrix[movie_id].mean()

    else:
        # Default to a rating of 3.0 in the absence of any information
        mean_rating = 3.0

    return mean_rating


# User Based Collaborative Filter using Weighted Mean Ratings
def cf_user_wmean(user_id, movie_id):
    # Check if movie_id exists in r_matrix
    if movie_id in r_matrix:

        # Get the similarity scores for the user in question with every other user
        sim_scores = cosine_sim[user_id]

        # Get the user ratings for the movie in question
        m_ratings = r_matrix[movie_id]

        # Extract the indices containing NaN in the m_ratings series
        idx = m_ratings[m_ratings.isnull()].index

        # Drop the NaN values from the m_ratings Series
        m_ratings = m_ratings.dropna()

        # Drop the corresponding cosine scores from the sim_scores series
        sim_scores = sim_scores.drop(idx)

        # Compute the final weighted mean
        wmean_rating = np.dot(sim_scores, m_ratings) / sim_scores.sum()

    else:
        # Default to a rating of 3.0 in the absence of any information
        wmean_rating = 3.0

    return wmean_rating


# Gender Based Collaborative Filter using Mean Ratings
def cf_gender(user_id, movie_id):
    # Check if movie_id exists in r_matrix (or training set)
    if movie_id in r_matrix:
        # Identify the gender of the user
        gender = users.loc[user_id]['sex']

        # Check if the gender has rated the movie
        if gender in gender_mean[movie_id]:

            # Compute the mean rating given by that gender to the movie
            gender_rating = gender_mean[movie_id][gender]

        else:
            gender_rating = 3.0

    else:
        # Default to a rating of 3.0 in the absence of any information
        gender_rating = 3.0

    return gender_rating


# Gender and Occupation Based Collaborative Filter using Mean Ratings
def cf_gen_occ(user_id, movie_id):
    # Check if movie_id exists in gen_occ_mean
    if movie_id in gen_occ_mean.index:

        # Identify the user
        user = users.loc[user_id]

        # Identify the gender and occupation
        gender = user['sex']
        occ = user['occupation']

        # Check if the occupation has rated the movie
        if occ in gen_occ_mean.loc[movie_id]:

            # Check if the gender has rated the movie
            if gender in gen_occ_mean.loc[movie_id][occ]:

                # Extract the required rating
                rating = gen_occ_mean.loc[movie_id][occ][gender]

                # Default to 3.0 if the rating is null
                if np.isnan(rating):
                    rating = 3.0

                return rating

    # Return the default rating
    return 3.0

if __name__ == "__main__":
    # Load the u.user file into a dataframe
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

    users = pd.read_csv(r'C:\datasets\ml-100k\u.user', sep='|', names=u_cols,
                        encoding='latin-1')

    print(users.head())

    # Load the u.items file into a dataframe
    i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    movies = pd.read_csv(r'C:\datasets\ml-100k\u.item', sep='|', names=i_cols, encoding='latin-1')

    print(movies.head())

    # Remove all information except Movie ID and title
    movies = movies[['movie_id', 'title']]

    # Load the u.data file into a dataframe
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    ratings = pd.read_csv(r'C:\datasets\ml-100k\u.data', sep='\t', names=r_cols,
                          encoding='latin-1')

    print("ratings head")
    print(ratings.head())

    # Drop the timestamp column
    ratings = ratings.drop('timestamp', axis=1)

    # Assign X as the original ratings dataframe and y as the user_id column of ratings.
    X = ratings.copy()
    y = ratings['user_id']

    # Split into training and test datasets, stratified along user_id
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    print(score(baseline))

    # Build the ratings matrix using pivot_table function
    r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='movie_id')

    print(r_matrix.head())

    print(score(cf_user_mean))

    # Create a dummy ratings matrix with all null values imputed to 0
    r_matrix_dummy = r_matrix.copy().fillna(0)

    # Compute the cosine similarity matrix using the dummy ratings matrix
    cosine_sim = cosine_similarity(r_matrix_dummy, r_matrix_dummy)

    # Convert into pandas dataframe
    cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.index, columns=r_matrix.index)

    print(cosine_sim.head(10))

    #print(score(cf_user_wmean))

    # Merge the original users dataframe with the training set
    merged_df = pd.merge(X_train, users)

    print(merged_df.head())

    # Compute the mean rating of every movie by gender
    gender_mean = merged_df[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()
    print(gender_mean)

    # Set the index of the users dataframe to the user_id
    users = users.set_index('user_id')
    print(score(cf_gender))

    # Compute the mean rating by gender and occupation
    gen_occ_mean = merged_df[['sex', 'rating', 'movie_id', 'occupation']].pivot_table(
        values='rating', index='movie_id', columns=['occupation', 'sex'], aggfunc='mean')

    print(gen_occ_mean.head())

    print(score(cf_gen_occ))

    # Define a Reader object
    # The Reader object helps in parsing the file or dataframe containing ratings
    reader = Reader()

    # Create the dataset to be used for building the filter
    data = Dataset.load_from_df(ratings, reader)

    # Define the algorithm object; in this case kNN
    knn = KNNBasic()

    # Evaluate the performance in terms of RMSE
    evaluate(knn, data, measures=['RMSE'])

    # Import SVD
    from surprise import SVD

    # Define the SVD algorithm object
    svd = SVD()

    # Evaluate the performance in terms of RMSE
    evaluate(svd, data, measures=['RMSE'])