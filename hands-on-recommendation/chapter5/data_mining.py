import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import pandas as pd
#Import the function that enables us to plot clusters
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import  pyplot as plt
#Import the K-Means Class
from sklearn.cluster import KMeans
import  seaborn as sns
#Import Standard Scaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Function to compute Euclidean Distance.
def euclidean(v1, v2):
    # Convert 1-D Python lists to numpy vectors
    v1 = np.array(v1)
    v2 = np.array(v2)
    # Compute vector which is the element wise square of the difference
    diff = np.power(np.array(v1) - np.array(v2), 2)

    # Perform summation of the elements of the above vector
    sigma_val = np.sum(diff)

    # Compute square root and return final Euclidean score
    euclid_score = np.sqrt(sigma_val)

    return euclid_score



if __name__ == "__main__":
    # Define 3 users with ratings for 5 movies
    u1 = [5, 1, 2, 4, 5]
    u2 = [1, 5, 4, 2, 1]
    u3 = [5, 2, 2, 4, 4]

    print(euclidean(u1, u2))

    print(euclidean(u1, u3))

    alice = [1, 1, 3, 2, 4]
    bob = [2, 2, 4, 3, 5]
    print(euclidean(alice, bob))
    eve = [5, 5, 3, 4, 2]
    print(euclidean(eve, alice))

    print(pearsonr(alice, bob))

    print(pearsonr(alice, eve))


    df = pd.DataFrame(index= ["alice", "bob", "eve"], data = [[1, 1, 3, 2, 4],[2, 2, 4, 3, 5],[5, 5, 3, 4, 2]])
    print(cosine_similarity(df,df))
    print(cosine_similarity(df,df)[1,2])

    # Get points such that they form 3 visually separable clusters
    X, y = make_blobs(n_samples=300, centers=3,
                      cluster_std=0.50, random_state=0)
    # Plot the points on a scatterplot
    plt.scatter(X[:, 0], X[:, 1], s=50)
    #plt.show()

    # Initializr the K-Means object. Set number of clusters to 3,
    # centroid initilalization as 'random' and maximum iterations to 10
    kmeans = KMeans(n_clusters=3, init='random', max_iter=10)

    # Compute the K-Means clustering
    kmeans.fit(X)

    # Predict the classes for every point
    y_pred = kmeans.predict(X)

    # Plot the data points again but with different colors for different classes
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50)

    # Get the list of the final centroids
    centroids = kmeans.cluster_centers_

    # Plot the centroids onto the same scatterplot.
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X')
    #plt.show()

    # List that will hold the sum of square values for different cluster sizes
    ss = []

    # We will compute SS for cluster sizes between 1 and 8.
    for i in range(1, 9):
        # Initialize the KMeans object and call the fit method to compute clusters
        kmeans = KMeans(n_clusters=i, random_state=0, max_iter=10, init='random').fit(X)

        # Append the value of SS for a particular iteration into the ss list
        ss.append(kmeans.inertia_)

    # Plot the Elbow Plot of SS v/s K
    sns.pointplot(x=[j for j in range(1, 9)], y=ss)
    #plt.show()

    # Import the half moon function from scikit-learn
    from sklearn.datasets import make_moons

    # Get access to points using the make_moons function
    X_m, y_m = make_moons(200, noise=.05, random_state=0)

    # Plot the two half moon clusters
    plt.scatter(X_m[:, 0], X_m[:, 1], s=50)

    # Initialize K-Means Object with K=2 (for two half moons) and fit it to our data
    kmm = KMeans(n_clusters=2, init='random', max_iter=10)
    kmm.fit(X_m)

    # Predict the classes for the data points
    y_m_pred = kmm.predict(X_m)

    # Plot the colored clusters as identified by K-Means
    plt.scatter(X_m[:, 0], X_m[:, 1], c=y_m_pred, s=50)

    #plt.show()

    # Import Spectral Clustering from scikit-learn
    from sklearn.cluster import SpectralClustering

    # Define the Spectral Clustering Model
    model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')

    # Fit and predict the labels
    y_m_sc = model.fit_predict(X_m)

    # Plot the colored clusters as identified by Spectral Clustering
    plt.scatter(X_m[:, 0], X_m[:, 1], c=y_m_sc, s=50)
    #plt.show()

    # Load the Iris dataset into Pandas DataFrame
    iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    # Display the head of the dataframe
    print(iris.head())

    # Separate the features and the class
    X = iris.drop('class', axis=1)
    y = iris['class']

    # Scale the features of X
    X = pd.DataFrame(StandardScaler().fit_transform(X),
                     columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    print(X.head())

    # Import PCA
    from sklearn.decomposition import PCA

    # Intialize a PCA object to transform into the 2D Space.
    pca = PCA(n_components=2)

    # Apply PCA
    pca_iris = pca.fit_transform(X)
    pca_iris = pd.DataFrame(data=pca_iris, columns=['PC1', 'PC2'])

    print(pca_iris.head())
    print(pca.explained_variance_ratio_)

    # Concatenate the class variable
    pca_iris = pd.concat([pca_iris, y], axis=1)

    # Display the scatterplot
    sns.lmplot(x='PC1', y='PC2', data=pca_iris, hue='class', fit_reg=False)
    plt.show()

    # Divide the dataset into the feature dataframe and the target class series.
    X, y = iris.drop('class', axis=1), iris['class']
    # Split the data into training and test datasets.
    # We will train on 75% of the data and assess our performance on 25% of the data

    # Import the splitting function
    from sklearn.model_selection import train_test_split

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Import the Gradient Boosting Classifier
    from sklearn.ensemble import GradientBoostingClassifier

    # Apply Gradient Boosting to the training data
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    # Display a bar plot of feature importances
    sns.barplot(x=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], y=gbc.feature_importances_)
    plt.show()
    # Compute the accuracy on the test set
    print(gbc.score(X_test, y_test))
