import os
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from itertools import combinations
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import traceback
import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp

# Set up basic logging
logging.basicConfig(level=logging.INFO)

path = os.getcwd()
# inputfile = os.path.join(path, 'top_5_recommendations.csv')
inputfile = os.path.join(path, 'top_5_recommendations_1.csv')
df = pd.read_csv(inputfile)
# df = df.head(2000)
print(df.shape)
print(df.head())
print(df.info())

def df_to_dict(df):
    """Convert a pandas DataFrame to a dictionary."""
    # Note: df.itterrows() is slow
    user_recommendations = {}
    for index, row in df.iterrows():
        user = row["User"]
        recommendations = {row["Rec1"], row["Rec2"], row["Rec3"], row["Rec4"], row["Rec5"]}
        user_recommendations[user] = recommendations
    return user_recommendations

def df_to_dict1(df):
    """Convert a pandas DataFrame to a dictionary."""
    user_recommendations = df.apply(lambda row: set(row), axis=1).to_dict()
    return user_recommendations

def jaccard_distance(set1, set2):
    """Calculate the Jaccard Distance between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union if union != 0 else 1

def calculate_personalization_index(recommendations):
    """Calculate the average Jaccard Distance across all user pairs."""
    distances = []
    for (user1, recs1), (user2, recs2) in combinations(recommendations.items(), 2):
        dist = jaccard_distance(recs1, recs2)
        distances.append(dist)

    return sum(distances) / len(distances) if distances else 0

# Define this as a top-level function
def compute_distance(recommendations, pair):
    try:
        user1, user2 = pair
        logging.info(f"Starting task {user1}-{user2}")
        return jaccard_distance(recommendations[user1], recommendations[user2])
    except Exception as e:
        traceback.print_exc()  # Print stack trace
        raise

def calculate_personalization_index_parallel(recommendations, n_workers=4):
    """Calculate the average Jaccard Distance across all user pairs using parallel processing."""
    user_pairs = list(combinations(recommendations.keys(), 2))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        from functools import partial
        func = partial(compute_distance, recommendations)
        distances = list(executor.map(func, user_pairs))

    return sum(distances) / len(distances) if distances else 0

def pairwise_jaccard_distances(df):
    """Calculate the Jaccard Distance using sklearn."""
    # if df has 'User' column, drop it
    if 'User' in df.columns:
        rec = df.drop('User', axis=1)
    else:
        rec = df
    
    # Flatten the DataFrame and find unique recommendations
    unique_recommendations = np.unique(rec.values.ravel())
    # Initiate a binary matrix with zeros
    binary_matrix = np.zeros((len(rec), len(unique_recommendations)), dtype=int)
    # Fill the binary matrix
    for i, row in df.iterrows():
        indices = np.searchsorted(unique_recommendations, row)
        binary_matrix[i, indices] = 1
    jaccard_distances = pairwise_distances(binary_matrix, metric="jaccard", n_jobs=-1)
    # Compute the average distance
    avg_distance = np.mean(jaccard_distances[np.triu_indices_from(jaccard_distances, k=1)])

    return avg_distance

def create_sparse_matrix(df):
    """Create a sparse CSR matrix from the DataFrame."""
    # Assuming the 'User' column exists and the rest are item recommendations
    if 'User' in df.columns:
        rec = df.drop('User', axis=1)
    else:
        rec = df

    # Flatten the DataFrame and find unique recommendations
    unique_recommendations = np.unique(rec.values.ravel())

    # Create row and column indices for the sparse matrix
    rows = np.repeat(np.arange(len(rec)), rec.shape[1])
    cols = np.searchsorted(unique_recommendations, rec.values.ravel())

    # Create a sparse matrix
    user_item_matrix = sp.csr_matrix((np.ones_like(rows), (rows, cols)), shape=(len(rec), len(unique_recommendations)))
    return user_item_matrix

def pairwise_jaccard_similarity_sparse(mat):
    """Calculate pairwise Jaccard similarity for a sparse matrix."""
    # Ensure the matrix is in CSR format
    mat_csr = mat.tocsr()

    # Calculate intersection and union counts
    intersection = mat_csr.dot(mat_csr.T)
    row_sums = mat_csr.sum(axis=1).A1
    union = row_sums[:, None] + row_sums - intersection

    # Calculate Jaccard similarity
    jaccard_similarity = intersection / union

    # Convert to Jaccard distance
    jaccard_distance = 1 - jaccard_similarity

    # Get the upper triangular part excluding the diagonal (i.e., unique pairs)
    triu_indices = np.triu_indices_from(jaccard_distance, k=1)
    avg_distance = np.mean(jaccard_distance[triu_indices])

    return avg_distance

# Example usage
# Assuming 'user_recommendations' is a dictionary where the key is the user ID and the value is a set of top 5 recommended items
# Example: {'user1': {'item1', 'item2', 'item3', 'item4', 'item5'}, 'user2': {'item6', 'item7', 'item3', 'item9', 'item10'}, ...}

if __name__ == "__main__":
    start_time = time.time()
    # user_recommendations = df_to_dict(df)
    # user_recommendations = df_to_dict1(df)
    # personalization_index = calculate_personalization_index(user_recommendations)
    # personalization_index = calculate_personalization_index_parallel(user_recommendations)
    personalization_index = pairwise_jaccard_distances(df)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Personalization Index: {personalization_index:.3f}")

    start_time = time.time()
    sparse_matrix = create_sparse_matrix(df)
    personalization_index = pairwise_jaccard_similarity_sparse(sparse_matrix)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Personalization Index: {personalization_index:.3f}")
 
 






def personalization_index(df):
    """
    INPUT:
    df - pandas dataframe with columns 'user_id', 'article_id', 'rank'
    
    OUTPUT:
    personalization_index - float that gives the personalization index of the dataframe
    
    Description:
    This function calculates the personalization index of a dataframe. The personalization index is a measure of how
    personalized the recommendations are for each user. It is calculated as the sum of the squared percentages of
    interactions for each article across all users. The lower the personalization index, the more personalized the
    recommendations are for each user.
    """
    # get the number of users
    num_users = df['user_id'].nunique()
    
    # get the number of articles
    num_articles = df['article_id'].nunique()
    
    # get the number of interactions
    num_interactions = df.shape[0]
    
    # get the number of interactions for each article
    article_interactions = df.groupby('article_id')['user_id'].count().reset_index()
    article_interactions.columns = ['article_id', 'num_interactions']
    
    # get the percentage of interactions for each article
    article_interactions['perc_interactions'] = article_interactions['num_interactions'] / num_interactions
    
    # get the sum of the squared percentages of interactions for each article
    sum_sq_perc_interactions = np.sum(article_interactions['perc_interactions']**2)
    
    # calculate the personalization index
    personalization_index = (1 - sum_sq_perc_interactions) / (1 - 1/num_articles)
    
    return personalization_index