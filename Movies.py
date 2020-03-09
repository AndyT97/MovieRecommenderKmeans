from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import csr_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score

#Setting output width
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

movies = pd.read_csv('movies.csv')
movies.head()
ratings = pd.read_csv('ratings.csv')
ratings.head()


# print('The dataset contains: ', len(ratings), ' ratings of ', len(movies), ' movies.')

# Function to get the genre ratings
def get_genre_ratings(ratings, movies, genres, column_names):
    genre_ratings = pd.DataFrame()
    for genre in genres:
        genre_movies = movies[movies['genres'].str.contains(genre)]
        avg_genre_votes_per_user =  ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)

    genre_ratings.columns = column_names
    return genre_ratings


# Calculate the average rating of romance and scifi movies
genre_ratings = get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
genre_ratings.head()


# Function to get the biased dataset
def bias_genre_rating_dataset(genre_ratings, score_limit_1, score_limit_2):
    biased_dataset = genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & (
            genre_ratings['avg_scifi_rating'] > score_limit_2)) | (
                                           (genre_ratings['avg_scifi_rating'] < score_limit_1) & (
                                           genre_ratings['avg_romance_rating'] > score_limit_2))]
    biased_dataset = pd.concat([biased_dataset[:300], genre_ratings[:2]])
    biased_dataset = pd.DataFrame(biased_dataset.to_records())
    return biased_dataset


# Bias the dataset
biased_dataset = bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

# Printing the resulting number of records & the head of the dataset
print("Number of records: ", len(biased_dataset))
biased_dataset.head()


# pprint(biased_dataset)

# Defining the scatterplot drawing function
def draw_scatterplot(x_data, x_label, y_data, y_label):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x_data, y_data, s=30)
    # plt.show()


# Plot the scatterplot
draw_scatterplot(biased_dataset['avg_scifi_rating'], 'Avg scifi rating', biased_dataset['avg_romance_rating'],
                 'Avg romance rating')

# Let's turn our dataset into a list
X = biased_dataset[['avg_scifi_rating', 'avg_romance_rating']].values

# Import KMeans
from sklearn.cluster import KMeans

# Create an instance of KMeans to find two clusters
kmeans_1 = KMeans(n_clusters=2)

# Use fit_predict to cluster the dataset
predictions = kmeans_1.fit_predict(X)


# Defining the cluster plotting function
def draw_clusters(biased_dataset, predictions, cmap='viridis'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel('Avg scifi rating')
    ax.set_ylabel('Avg romance rating')

    clustered = pd.concat([biased_dataset.reset_index(), pd.DataFrame({'group': predictions})], axis=1)
    plt.scatter(clustered['avg_scifi_rating'], clustered['avg_romance_rating'], c=clustered['group'], s=20, cmap=cmap)
    plt.show()


# Plot
# draw_clusters(biased_dataset, predictions)


# Create an instance of KMeans to find three clusters
kmeans_2 = KMeans(n_clusters=7)
# Use fit_predict to cluster the dataset
predictions_2 = kmeans_2.fit_predict(X)

# Plot After drawing this cluster we can tell people who like sci fi belong to the yellow group
# People who like romance belong to the purple group and people who like both belong to green
#draw_clusters(biased_dataset, predictions_2)


#TODO Algorithm to find number of clusters not working

# # Choosing the right number of clusters :
# # We will use the elbow method,
#
# # Selecting our dataset to study
# df = biased_dataset[['avg_scifi_rating', 'avg_romance_rating']]
#
# # Choose the range of k values to test.
# # We added a stride of 5 to improve performance. We don't need to calculate the error for every k value
# possible_k_values = range(2, len(X) + 1, 2)
#
#
# # Define function to calculate the clustering errors
# def clustering_errors(k, data):
#     kmeans = KMeans(n_clusters=k).fit(data)
#     predictions = kmeans.predict(data)
#     # cluster_centers = kmeans.cluster_centers_
#     # errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data.values, predictions)]
#     # return sum(errors)
#     silhouette_avg = silhouette_score(data, predictions)
#     return silhouette_avg
#
#
# # Calculate error values for all k values we're interested in
# errors_per_k = [clustering_errors(k, X) for k in possible_k_values]
#
# # Plot the each value of K vs. the silhouette score at that value
# fig, ax = plt.subplots(figsize=(16, 6))
# plt.plot(possible_k_values, errors_per_k)
# plt.show()
#
# # Ticks and grid
# xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 2)
# ax.set_xticks(xticks, minor=False)
# ax.set_xticks(xticks, minor=True)
# ax.xaxis.grid(True, which='both')
# yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
# ax.set_yticks(yticks, minor=False)
# ax.set_yticks(yticks, minor=True)
# ax.yaxis.grid(True, which='both')


biased_dataset_3_genres = get_genre_ratings(ratings, movies, ['Romance','Sci-Fi','Action'],['avg_romance_rating','avg_scifi_rating','avg_action_rating'])

#Score limit 1 and score limit 2
biased_dataset_3_genres = bias_genre_rating_dataset(biased_dataset_3_genres, 3.2,2.5).dropna()

print ("Number of records including action: ", len(biased_dataset_3_genres))
print(biased_dataset_3_genres.head())

#Turn dataset into a list
xAction = biased_dataset_3_genres[['avg_scifi_rating','avg_romance_rating','avg_action_rating']].values

#Use 7 clusters
kmeans_3 = KMeans(n_clusters=7)

#Use fit_predict to cluster the dataset
predictions_3 = kmeans_3.fit_predict(xAction)

#Plot
def draw_clusters_3d(biased_dataset_3, predictions):

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel('Avg scifi rating')
    ax.set_xlabel('Avg romance rating')

    clustered = pd.concat([biased_dataset_3.reset_index(),pd.DataFrame({'group':predictions})], axis=1)

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for g in clustered.group.unique():
        color = next(colors)
        for index, point in clustered[clustered.group ==g].iterrows():
            if point['avg_action_rating'].astype(float) > 3:
                size = 50
            else:
                size = 15
            plt.scatter(point['avg_scifi_rating'],point['avg_romance_rating'],s=size,color = color)

    plt.show()


#Uncomment this
#draw_clusters_3d(biased_dataset_3_genres,predictions_3)


# Higher-Level Clustering

#Merge the two tables then pivot so we have Users X Movies dataframe
ratings_title = pd.merge(ratings, movies[['movieId','title']], on='movieId')
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns='title', values='rating')

# Print the number of dimensions and a subset of the dataset
#print('dataset dimensions: ', user_movie_ratings.shape, '\n\nSubset example:')
#After printing this we see a lot of NaN values because most users have not rated most of the movies
#We will sort the dataset by the most rated movies
#print(user_movie_ratings.iloc[:6, :10])

#Define function to get most rated movies

def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
    #1 - Count
    user_movie_ratings = user_movie_ratings.append(user_movie_ratings.count(), ignore_index = True)
    #2 - Sort
    user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings) - 1, axis=1, ascending=False)
    user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    #3 - Slice
    most_rated_movies = user_movie_ratings_sorted.iloc[:max_number_of_movies]
    return most_rated_movies


#Sorting by rating function
def sort_by_rating_density (user_movie, n_movies, n_users):
    most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
    #TODO Define function
    #most_rated_movies = get_users_who_rate_most(most_rated_movies,n_users)
    return most_rated_movies

n_movies = 30
n_users = 18
most_rated_movies_users_selection = sort_by_rating_density(user_movie_ratings, n_movies, n_users)

#print('dataset dimensions')
#most_rated_movies_users_selection.shape()
#print(most_rated_movies_users_selection.head())

#Pivot the dataset and choose the first 1000 movies
user_movie_ratings =  pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
most_rated_movies_1k = get_most_rated_movies(user_movie_ratings, 1000)

#sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())

# 20 clusters
#predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)
# Select the mas number of users and movies heatmap cluster
max_users = 70
max_movies = 50

# Cluster and print some of them
clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
#draw_movie_clusters(clustered, max_users, max_movies)

# # Pick a cluster ID from the clusters above
cluster_number = 11
# Let's filter to only see the region of the dataset with the most number of values
n_users = 75
n_movies = 300
cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)
# Sort and print the cluster
cluster = sort_by_rating_density(cluster, n_movies, n_users)
#draw_movies_heatmap(cluster, axis_labels=False)
#
# # Fill in the name of the column/movie. e.g. 'Forrest Gump (1994)'
# movie_name = "Matrix, The (1999)"
# cluster[movie_name].mean()

# Pick a user ID from the dataset
user_id = 10
# Get all this user's ratings
user_2_ratings  = cluster.loc[user_id, :]
# Which movies did they not rate?
user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]
# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]
# Let's sort by rating so the highest rated movies are presented first
print(avg_ratings.sort_values(ascending=False)[:20])