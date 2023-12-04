import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_age_distribution(users_df):
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

def plot_book_year_distribution(items_df):
    # book publication year distribution
    sns.displot(items_df['Year-Of-Publication'], kde=False, bins=range(1940, 2011, 5), height=6, aspect=2)
    plt.title('Book Publication Year Distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Year of publication')
    plt.xticks(range(1940, 2011, 5))
    plt.xlim(1940, 2011)
    plt.show()

def plot_rating_distribution(ratings_df):
    # plot the distribution of ratings
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=ratings_df)
    plt.title('Distribution of Ratings')
    plt.show()


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