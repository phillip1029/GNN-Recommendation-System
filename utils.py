import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_data(items_df, users_df, ratings_df, all_ratings_df):
    # # Save current data for later use
    with open('data/item.pkl', 'wb') as f:
        pickle.dump(items_df, f)

    with open('data/user.pkl', 'wb') as f:
        pickle.dump(users_df, f)

    with open('data/rating.pkl', 'wb') as f:
        pickle.dump(ratings_df, f)

    with open('data/all_rating.pkl', 'wb') as f:
        pickle.dump(all_ratings_df, f)

def load_data():
    # Load the object from the file
    with open('data/item.pkl', 'rb') as f:
        items_df = pickle.load(f)

    with open('data/user.pkl', 'rb') as f:
        users_df = pickle.load(f)

    with open('data/rating.pkl', 'rb') as f:
        ratings_df = pickle.load(f)

    with open('data/all_rating.pkl', 'rb') as f:
        all_ratings_df = pickle.load(f)

    return items_df, users_df, ratings_df, all_ratings_df

def plot_age_distribution(users_df):
    # Plot the age distribution
    plt.figure(figsize=(12, 8))
    plt.hist(users_df['age'], bins=20, color='orange')
    plt.title('Age Distribution', fontsize=20)
    plt.xlabel('Age', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()