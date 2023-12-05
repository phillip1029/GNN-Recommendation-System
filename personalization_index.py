import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
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

# def not_used():
#     def df_to_dict(df):
#         """Convert a pandas DataFrame to a dictionary."""
#         # Note: df.itterrows() is slow
#         user_recommendations = {}
#         for index, row in df.iterrows():
#             user = row["User"]
#             recommendations = {row["Rec1"], row["Rec2"], row["Rec3"], row["Rec4"], row["Rec5"]}
#             user_recommendations[user] = recommendations
#         return user_recommendations

#     def df_to_dict1(df):
#         """Convert a pandas DataFrame to a dictionary."""
#         user_recommendations = df.apply(lambda row: set(row), axis=1).to_dict()
#         return user_recommendations

#     def jaccard_distance(set1, set2):
#         """Calculate the Jaccard Distance between two sets."""
#         intersection = len(set1.intersection(set2))
#         union = len(set1.union(set2))
#         return 1 - intersection / union if union != 0 else 1

#     def calculate_personalization_index(recommendations):
#         """Calculate the average Jaccard Distance across all user pairs."""
#         distances = []
#         for (user1, recs1), (user2, recs2) in combinations(recommendations.items(), 2):
#             dist = jaccard_distance(recs1, recs2)
#             distances.append(dist)

#         return sum(distances) / len(distances) if distances else 0

#     # Define this as a top-level function
#     def compute_distance(recommendations, pair):
#         try:
#             user1, user2 = pair
#             logging.info(f"Starting task {user1}-{user2}")
#             return jaccard_distance(recommendations[user1], recommendations[user2])
#         except Exception as e:
#             traceback.print_exc()  # Print stack trace
#             raise

#     def calculate_personalization_index_parallel(recommendations, n_workers=4):
#         """Calculate the average Jaccard Distance across all user pairs using parallel processing."""
#         user_pairs = list(combinations(recommendations.keys(), 2))

#         with ProcessPoolExecutor(max_workers=n_workers) as executor:
#             from functools import partial
#             func = partial(compute_distance, recommendations)
#             distances = list(executor.map(func, user_pairs))

#         return sum(distances) / len(distances) if distances else 0
# # end of not_used()

# ref: https://stackoverflow.com/questions/71554288/efficient-pairwise-jaccard-score-with-two-dataframes
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

# below two functions are to use sparse matrix method
# ref: https://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html
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

    # Avoid division by zero
    union = np.where(union != 0, union, 1)

    # Calculate Jaccard similarity
    jaccard_similarity = intersection / union

    # Convert to Jaccard distance
    jaccard_distance = (1 - jaccard_similarity).toarray()

    # Get the upper triangular part excluding the diagonal (i.e., unique pairs)
    triu_indices = np.triu_indices_from(jaccard_distance, k=1)
    avg_distance = np.mean(jaccard_distance[triu_indices])

    return avg_distance

def simpson_diversity_index(df):
    """Calculate the Simpson Diversity Index for a given list of recommendations.
    D = 1 - (sum((n/N)^2))
    """
    # Assuming the 'User' column exists and the rest are item recommendations
    if 'User' in df.columns:
        rec = df.drop('User', axis=1)
    else:
        rec = df
    # Flatten the DataFrame into a single list of all recommendations
    combined_recommendations = df.values.flatten()
    # Count the occurrences of each item
    count = Counter(combined_recommendations)
    N = sum(count.values())
    diversity_index = 1 - sum((n / N) ** 2 for n in count.values())

    return diversity_index

def plot_item_percentage_bar(df):
    # Flatten the DataFrame to get a list of all recommendations
    all_recommendations = df.values.flatten()

    # Count the frequency of each item in the recommendations
    item_frequency = Counter(all_recommendations)

    # Convert to a DataFrame for plotting
    item_freq_df = pd.DataFrame(item_frequency.items(), columns=['Item', 'Frequency'])

    # Calculate the percentage of total for each item's frequency
    total_recommendations = len(all_recommendations)
    item_freq_df['Percentage'] = (item_freq_df['Frequency'] / total_recommendations) * 100

    # Sorting the DataFrame by the percentage in descending order
    item_freq_df_sorted_by_percentage = item_freq_df.sort_values(by='Percentage', ascending=False, ignore_index=True)

    # Plotting with the correctly sorted data
    plt.figure(figsize=(20, 6))
    # Generate a color palette with a color for each bar
    n = len(item_freq_df_sorted_by_percentage)
    palette = sns.color_palette("viridis", n_colors=n)
    # Plot using the generated palette
    sns.barplot(x=item_freq_df_sorted_by_percentage.index, y='Percentage', data=item_freq_df_sorted_by_percentage, palette=palette)
    plt.xlabel('Item Index (Sorted by Percentage)')
    plt.ylabel('Percentage of Total Recommendations (%)')
    plt.title('Percentage of Total Recommendations for Each Item (Sorted by Percentage)')
    plt.xticks([])
    plt.show()

def plot2_item_percentage_bar(recommendations_df, top=True, cumulative=False):
    # Flatten the DataFrame to get a list of all recommendations
    all_recommendations = recommendations_df.values.flatten()

    # Count the frequency of each item in the recommendations
    item_frequency = Counter(all_recommendations)

    # Convert to a DataFrame for plotting
    item_freq_df = pd.DataFrame(item_frequency.items(), columns=['Item', 'Frequency'])

    # Calculate the percentage of total for each item's frequency
    total_recommendations = len(all_recommendations)
    item_freq_df['Percentage'] = (item_freq_df['Frequency'] / total_recommendations) * 100

    # Sorting the DataFrame by the percentage in descending order
    item_freq_df_sorted_by_percentage = item_freq_df.sort_values(by='Percentage', ascending=False, ignore_index=True)

    # Calculating the cumulative percentage if required
    if cumulative:
        item_freq_df_sorted_by_percentage['Cumulative_Percentage'] = item_freq_df_sorted_by_percentage['Percentage'].cumsum()

    # Selecting the top N items
    if top:
        top_N = 60
        item_freq_df_sorted_by_percentage = item_freq_df_sorted_by_percentage.head(top_N)

    # Plotting with the correctly sorted data
    plt.figure(figsize=(20, 6))

    # Choose the column to plot
    y_column = 'Cumulative_Percentage' if cumulative else 'Percentage'

    # Generate a color palette with a color for each bar
    n = len(item_freq_df_sorted_by_percentage)
    palette = sns.color_palette("viridis", n_colors=n)

    # Plot using the generated palette
    sns.barplot(x=item_freq_df_sorted_by_percentage.index, y=y_column, data=item_freq_df_sorted_by_percentage, palette=palette)
    plt.xlabel('Item Index')
    plt.ylabel(f'{"Cumulative " if cumulative else ""}Percentage of Total Recommendations (%)')
    plt.title(f'{"Cumulative " if cumulative else ""}Percentage of Total Recommendations for Each Item')
    if top:
        plt.xticks(range(top_N), item_freq_df_sorted_by_percentage['Item'], rotation=45)
    else:
        plt.xticks([])  # Hides the x-axis labels
    plt.show()

# def plot2_item_percentage_bar(recommendations_df, top=True):
#     # Flatten the DataFrame to get a list of all recommendations
#     all_recommendations = recommendations_df.values.flatten()

#     # Count the frequency of each item in the recommendations
#     item_frequency = Counter(all_recommendations)

#     # Convert to a DataFrame for plotting
#     item_freq_df = pd.DataFrame(item_frequency.items(), columns=['Item', 'Frequency'])

#     # Calculate the percentage of total for each item's frequency
#     total_recommendations = len(all_recommendations)
#     item_freq_df['Percentage'] = (item_freq_df['Frequency'] / total_recommendations) * 100

#     # Sorting the DataFrame by the percentage in descending order
#     item_freq_df_sorted_by_percentage = item_freq_df.sort_values(by='Percentage', ascending=False, ignore_index=True)

#     # Selecting the top N items
#     if top:
#         top_N = 60
#         item_freq_df_sorted_by_percentage = item_freq_df_sorted_by_percentage.head(top_N)

#     # Plotting with the correctly sorted data
#     plt.figure(figsize=(20, 6))

#     # Generate a color palette with a color for each bar
#     n = len(item_freq_df_sorted_by_percentage)
#     palette = sns.color_palette("viridis", n_colors=n)

#     # Plot using the generated palette and hue parameter
#     sns.barplot(x=item_freq_df_sorted_by_percentage.index, y='Percentage', data=item_freq_df_sorted_by_percentage, 
#                 hue=item_freq_df_sorted_by_percentage.index, palette=palette, dodge=False)
#     plt.xlabel('Item Index')
#     plt.ylabel('Percentage of Total Recommendations (%)')
#     plt.title('Percentage of Total Recommendations for Each Item (Sorted by Percentage)')
#     if top:
#         plt.xticks(range(top_N), item_freq_df_sorted_by_percentage['Item'], rotation=45)
#     else:
#         plt.xticks([])  # Hides the x-axis labels
#     plt.legend([],[], frameon=False)  # Hide the legend
#     plt.show()

def load_result():
    df = pd.read_csv('data/top_5_recommendations.csv')
    return df

def plot_same_recommendations(df):
    # Function to get all combination counts for a given size
    def get_combination_counts(data, size):
        comb_counts = Counter()
        for row in data.itertuples(index=False):
            for comb in combinations(row, size):
                sorted_comb = tuple(sorted(comb))
                comb_counts[sorted_comb] += 1
        return comb_counts

    # Storing combination counts for sizes 2, 3, 4, 5
    combination_counts = {size: get_combination_counts(df, size) for size in range(0, 6)}

    # Counting how many users received the same combinations
    same_comb_users = {size: sum(1 for count in comb_count.values() if count > 1) 
                    for size, comb_count in combination_counts.items()}

    # Calculate the percentage of total users for each combination size
    total_users = len(df)
    percent_same_comb_users = {size: (count / total_users) * 100 for size, count in same_comb_users.items()}

    # Convert to a DataFrame for plotting
    percent_same_comb_users_df = pd.DataFrame(list(percent_same_comb_users.items()), columns=['Combination Size', 'Percentage of Users'])

    # Plotting with percentage labels on top of each bar
    plt.figure(figsize=(10, 6))
    bars = plt.bar(percent_same_comb_users_df['Combination Size'], percent_same_comb_users_df['Percentage of Users'])

    # Add labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{round(yval)}%', ha='center', va='bottom')

    plt.xlabel('Combination Size')
    plt.ylabel('Percentage of Users (%)')
    plt.title('Percentage of Users Receiving Same Top 5 Recommendations')
    plt.xticks(range(2, 6))  # Set x-ticks to be the combination sizes
    plt.show()

if __name__ == "__main__":
    path = os.getcwd()
    inputfile = os.path.join(path, 'data/top_5_recommendations.csv')
    df = pd.read_csv(inputfile)
    # df = df.head(2000)
    print(df.shape)
    print(df.head())
    print(df.info())

    # user_recommendations = df_to_dict(df)
    # user_recommendations = df_to_dict1(df)
    # personalization_index = calculate_personalization_index(user_recommendations)
    # personalization_index = calculate_personalization_index_parallel(user_recommendations)

    start_time = time.time()
    personalization_index = pairwise_jaccard_distances(df)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Personalization Index: {personalization_index:.4f}")
    # Execution time: 495.508 seconds
    # Personalization Index: 0.953

    start_time = time.time()
    sparse_matrix = create_sparse_matrix(df)
    personalization_index = pairwise_jaccard_similarity_sparse(sparse_matrix)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Personalization Index: {personalization_index:.4f}")
    # Execution time: 435.011 seconds
    # Personalization Index: 0.953

    plot_item_percentage_bar(df)

    plot2_item_percentage_bar(df)

    start_time = time.time()
    simpson_diversity_index = simpson_diversity_index(df)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Simpson Diversity Index: {simpson_diversity_index:.4f}")
    # Execution time: 0.064 seconds
    # Simpson Diversity Index: 0.9329
    
 







# def personalization_index(df):
#     """
#     INPUT:
#     df - pandas dataframe with columns 'user_id', 'article_id', 'rank'
    
#     OUTPUT:
#     personalization_index - float that gives the personalization index of the dataframe
    
#     Description:
#     This function calculates the personalization index of a dataframe. The personalization index is a measure of how
#     personalized the recommendations are for each user. It is calculated as the sum of the squared percentages of
#     interactions for each article across all users. The lower the personalization index, the more personalized the
#     recommendations are for each user.
#     """
#     # get the number of users
#     num_users = df['user_id'].nunique()
    
#     # get the number of articles
#     num_articles = df['article_id'].nunique()
    
#     # get the number of interactions
#     num_interactions = df.shape[0]
    
#     # get the number of interactions for each article
#     article_interactions = df.groupby('article_id')['user_id'].count().reset_index()
#     article_interactions.columns = ['article_id', 'num_interactions']
    
#     # get the percentage of interactions for each article
#     article_interactions['perc_interactions'] = article_interactions['num_interactions'] / num_interactions
    
#     # get the sum of the squared percentages of interactions for each article
#     sum_sq_perc_interactions = np.sum(article_interactions['perc_interactions']**2)
    
#     # calculate the personalization index
#     personalization_index = (1 - sum_sq_perc_interactions) / (1 - 1/num_articles)
    
#     return personalization_index