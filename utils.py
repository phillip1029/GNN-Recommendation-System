import pickle

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