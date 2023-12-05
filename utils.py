import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests

# from geopy.geocoders import Nominatim
# import folium

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

def plot_age_distribution():
    _, users_df, _, _ = load_data()
    # Plotting the age distribution
    plt.figure(figsize=(10, 6))
    plt.hist(users_df["age"], bins=range(0, 100, 5), edgecolor='black')
    plt.title("Age Distribution of Users")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.xticks(range(0, 100, 5))
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# def plot_age_group_distribution(users_df):
#     # Count the frequency of each category
#     labels = ['0-18', '19-35', '36-55', '56-75', '76+']
#     age_group_counts = users_df['age_group'].value_counts().reindex(labels)

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     age_group_counts.plot(kind='bar')
#     plt.title("Age Group Distribution of Users")
#     plt.xlabel("Age Group")
#     plt.ylabel("Frequency")
#     plt.show()

def plot_book_year_distribution():
    items_df, _, _, _ = load_data()
    # book publication year distribution
    sns.displot(items_df['Year-Of-Publication'], kde=False, bins=range(1940, 2011, 5), height=6, aspect=2)
    plt.title('Book Publication Year Distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Year of publication')
    plt.xticks(range(1940, 2011, 5))
    plt.xlim(1940, 2011)
    plt.show()

def plot_rating_distribution():
    _, _, _, all_ratings_df = load_data()
    # plot the distribution of ratings
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=all_ratings_df)
    plt.title('Distribution of Ratings')
    plt.show()

def recommend_items_to_user(user_id):
    """
    This function takes in a user id and returns a list of 5 recommended items for that user
    """
    top_5_recommendations_long = pd.read_csv('data/top_5_recommendations_long.csv')
    items_df = pd.read_pickle('data/items_df.pkl')
    all_ratings_df = pd.read_pickle('data/all_ratings_df.pkl')

    # get all the rows of the user from ratings_df
    top_recommends = top_5_recommendations_long.loc[top_5_recommendations_long['user_id'] == user_id]
    # merge with items_df to get all other item info
    top_recommends = top_recommends.merge(items_df, on='item_idx', how='left')
     # all the title
    titles = top_recommends['title'].tolist()
    # exclude rows where rating less than 1 in all_ratings_df
    item_ratings = all_ratings_df.loc[all_ratings_df['rating'] >= 0]

    # print the top 5 recommendations: {title} by {author} published in {Year-Of-Publication}
    print(f"Top 5 recommendations for user {user_id}:")
    for i in range(5):
        print(f"{i+1}. {top_recommends.iloc[i]['title']} by {top_recommends.iloc[i]['author']} published in {top_recommends.iloc[i]['Year-Of-Publication']}")
   
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    fig, axs = plt.subplots(1, 5, figsize=(20,6))
    fig.patch.set_alpha(0)
    for i, title in enumerate(titles):        
        url = items_df.loc[items_df['title'] == title]['Image-URL-L'][:1].values[0]
        img = Image.open(requests.get(url, stream=True, headers=headers).raw)
        rating = item_ratings.loc[(item_ratings['item_id'] == items_df.loc[items_df['title'] == title]['item_id'][:1].values[0]) & (item_ratings['rating'] != 0)]['rating'].mean()

        # rating = item_ratings.loc[item_ratings['item_id'] == items_df.loc[items_df['title'] == title]['item_id'][:1].values[0]]['rating'].mean()
        axs[i].axis("off")
        axs[i].imshow(img)
        axs[i].set_title(f'{rating:.1f}/10', y=-0.1, fontsize=18)

    # print the top 5 recommendations: {title} by {author} published in {Year-Of-Publication}
    # print a line of break
    print("----------------------------------------------------------------------------------------")
    print(f"Top favorite items for user {user_id}:")
    rating_history = item_ratings[item_ratings['user_id'] == user_id]
    top_history = rating_history.sort_values(by=['rating'], ascending=False)
    top_history = pd.merge(top_history, items_df, on='item_id', how='inner')
    for i in range(len(top_history)):
        print(f"{i+1}. {top_history.iloc[i]['title']} by {top_history.iloc[i]['author']} published in {top_history.iloc[i]['Year-Of-Publication']}")


# def get_lat_long(df_in):
#     # Not Working for geo location
#     df = df_in.copy()
#     locations = df['location'].unique()  # unique locations

#     geolocator = Nominatim(user_agent="your_app_name")

#     # Geocoding each location
#     latitudes = []
#     longitudes = []
#     for loc in locations:
#         location = geolocator.geocode(loc)
#         if location:
#             latitudes.append(location.latitude)
#             longitudes.append(location.longitude)
#         else:
#             latitudes.append(None)
#             longitudes.append(None)

#     # Adding the coordinates to the dataframe
#     df['latitude'] = df['location'].map(dict(zip(locations, latitudes)))
#     df['longitude'] = df['location'].map(dict(zip(locations, longitudes)))
    
#     return df

# def plot_geo(users_df):
#     # Plotting the locations of users
#     users_df_plot = get_lat_long(users_df)

#     # Create a base map
#     average_latitude = users_df_plot['latitude'].mean()
#     average_longitude = users_df_plot['longitude'].mean()
#     m = folium.Map(location=[average_latitude, average_longitude], zoom_start=6)

#     # Add points for each location
#     for index, row in users_df_plot.dropna(subset=['latitude', 'longitude']).iterrows():
#         folium.Marker([row['latitude'], row['longitude']], popup=row['location']).add_to(m)

#     # Display the map
#     m