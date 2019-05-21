# Config.py

class Directories:

    #Directories
    d_raw = "C:/datasets/the-movies-dataset/"
    d_prep = "C:/datasets/the-movies-dataset/"
    d_model = "C:/datasets/the-movies-dataset/models/"
    d_model_coll = "C:/datasets/the-movies-dataset/models/"
    d_model_content = "C:/datasets/the-movies-dataset/models/content_based/"
    d_model_hybrid = "C:/datasets/the-movies-dataset/models/collaborative_based/"

class Files:
    #Raw File Names
    f_movies_metadata = Directories.d_raw + "movies_metadata.csv"
    f_movie_ids = Directories.d_raw + "movie_ids.csv"
    f_keywords = Directories.d_raw + "keywords.csv"
    f_credits = Directories.d_raw + "credits.csv"
    f_ratings = Directories.d_raw + "ratings.csv"
    f_users = Directories.d_raw + "users.csv"
    f_user_demographic = Directories.d_raw + "u.user"

    #Prep File Names
    fp_movies_metadata = Directories.d_prep + "prep_data.csv"
    fp_ratings = Directories.d_prep + "prep_ratings.csv"

    # Model File Prefixes
    p_mf_coll = "model_coll_"
    p_mf_cont = "model_content_"

    # Model File Suffixes
    s_mf_keys = "_keys"

class Attributes:
    c_userId = "userId"
    c_contentId = "id"
    c_rating = "rating"