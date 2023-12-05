# import libraries
from pyvis.network import Network  # Interactive network visualization
import pandas as pd # Data Frame
import networkx as nx # Network analysis
from tqdm.notebook import tqdm # Progress bar
import webbrowser
# Read data
ratings_df = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', encoding='latin-1')
users_df = pd.read_csv('data/BX-Users.csv', sep=';', encoding='latin-1')
books_df = pd.read_csv('data/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
books_df.columns = books_df.columns.str.lower()
books_df.columns = books_df.columns.str.replace('-','_')
users_df.columns = users_df.columns.str.lower()
users_df.columns = users_df.columns.str.replace('-','_')
ratings_df.columns = ratings_df.columns.str.lower()
ratings_df.columns = ratings_df.columns.str.replace('-','_')
ratings_df = ratings_df[ratings_df['book_rating'] > 0]
def getColor(num):
    selection = num % 4
    table = {
        0:"#611C35",
        1:"#2E5077",
        2:"#FFA630",
        3:"#FFD700",
    }
    return table[selection]

def getSize(num):
    selection = num % 4
    table = {
        0:600,
        1:400,
        2:200,
        3:100,
    }
    return table[selection]

def processing(items:set,next_layer_df:pd.DataFrame,current_layer_column_name:str, next_layer_column_name:str,layer_count:int,G:nx.Graph,next_item_df:pd.DataFrame,top_n:int=5):
    temp_set = set()
    for item in tqdm(items, desc=f"Processing current item, layer {layer_count}"):
        # next_layer_items = next_layer_df[next_layer_df[current_layer_column_name] == item][next_layer_column_name].unique()
        sorted_items = next_layer_df.sort_values(by='book_rating', ascending=False)
        filtered_sorted_items = sorted_items[sorted_items[current_layer_column_name] == item]
        # Get unique values and pick top 5
        top_5_next_layer_items = filtered_sorted_items[next_layer_column_name].unique()[:top_n]
        for next_layer_item in top_5_next_layer_items:
            test_name = f"{layer_count-2}_{next_layer_column_name}:{next_layer_item}"
            if G.has_node(test_name):
                print(f"Node already exists: {test_name}")
            else:
                temp_set.add(next_layer_item)
                try:
                    base_name = f"{layer_count}_{next_layer_column_name}:{next_layer_item}"
                    next_item = next_item_df[next_item_df[next_layer_column_name]==next_layer_item].iloc[0]
                    title = ''
                    if(next_layer_column_name=='isbn'):
                        title = f"layer:{layer_count}\nisbn:{next_item['isbn']}\nbook_title:{next_item['book_title']}\nbook_author:{next_item['book_author']}"
                    else:
                        title = f"layer:{layer_count}\nuser_id:{next_item['user_id']}\nlocation:{next_item['location']}\nage:{next_item['age']}"
                    G.add_node(base_name,title=title, color=getColor(layer_count), size=getSize(layer_count))
                    G.add_edge(f"{layer_count-1}_{current_layer_column_name}:{item}",base_name)
                except IndexError:
                    print(f"IndexError: {next_layer_item}")
                    temp_set.discard(next_layer_item)
    return temp_set

def buildLayers(_target_user_id, _layer_count, _df, _G:nx.Graph,top_n:int=5):
    root_user = users_df[users_df['user_id']==_target_user_id].iloc[0]
    title = f"0_user_id:{root_user['user_id']}\nlocation:{root_user['location']}\nage:{root_user['age']}"
    _G.add_node(f"0_user_id:{_target_user_id}", label=str(_target_user_id), color="#611C35",size = getSize(0),title=title)
    # Building the tree
    current_layer_users = set()
    current_layer_users.add(_target_user_id)
    for layer in tqdm(range(1, _layer_count + 1), desc="Building the layer"):
        if layer % 2 == 1:  # Odd layers: find books
            current_layer_users = processing(current_layer_users, _df, 'user_id', 'isbn', layer, _G, books_df,top_n)
        else:  # Even layers: find users
            current_layer_users = processing(current_layer_users, _df, 'isbn', 'user_id', layer, _G, users_df,top_n)

def draw_user_graph(target_user_id:int,layer_count:int,top_n:int=5):
    # Initialize Pyvis Network
    net = Network(select_menu=True)
    # net.show_buttons(filter_=['physics'])
    net.barnes_hut()
    G = nx.Graph()

    buildLayers(target_user_id,layer_count,ratings_df,G,top_n=top_n)
    print(f"Number of nodes: {len(G.nodes)}")
    print(f"Number of edges: {len(G.edges)}")
    net.from_nx(G)
    # Display the network
    filename = f'user_book_network_{target_user_id}.html'
    net.write_html(filename)
    file = open(filename,'r')
    content = file.read()
    content = content.replace('id="mynetwork"', 'id="mynetwork" style="height: 100dvh;"')
    file.close()
    file = open(filename,'w')
    file.write(content)
    file.close()
    webbrowser.open(f'{filename}')


if __name__ == '__main__':
    draw_user_graph(236172,3,10)
    draw_user_graph(236179,3,10)
    draw_user_graph(236184,3,10)
    draw_user_graph(236198,3,10)