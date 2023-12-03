import networkx as nx
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt

#credentials
CLIENT_ID = " // "
CLIENT_SECRET = "//"
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)


#The function will receive the name of the artist to search for and will return a string with the Spotify ID (Spotify ID) of this artist.
def search_artist(artist_name: str) -> str:
    """
    Search for an artist in Spotify.
    :param artist_name: name to search for.
    :return: spotify artist id.
    """
    results = sp.search(q=artist_name, type='artist')
    return results['artists']['items'][0]['id']

'''
Il crawler prenderà come seme l'identificativo di un artista e otterrà i suoi artisti correlati, che saranno utilizzati per continuare il processo di acquisizione dei dati.
Devono essere implementati due algoritmi di schedulazione: breadth-first search (BFS) e depth-first search (DFS). Per entrambi gli algoritmi, nel caso in cui si debba decidere quale nodo esplorare tra un insieme di nodi dello stesso livello (cioè nel caso in cui si debba rompere un pareggio), verrà sempre selezionato il primo artista ottenuto dall'API.
Il crawler si ferma quando ha esplorato il numero massimo di artisti indicato come parametro o quando non ci sono più artisti noti da esplorare. (esplorare un artista significa recuperare tutti gli artisti correlati)
Il crawler creerà un grafo utilizzando networkx con i dati ottenuti (i nodi rappresenteranno gli artisti e gli spigoli le relazioni tra gli artisti). forniti dalla chiamata "artista correlato" di Spotify). Per ogni artista, memorizzeremo almeno il nome, l'identificativo Spotify, il numero di follower, la popolarità e i generi musicali associati.
Infine, la funzione salverà il grafo in formato graphml nel file indicato come parametro e restituirà il grafo (l'oggetto grafo networkx).'''

def crawler(seed: str, max_nodes_to_crawl: int, strategy: str = "BFS", out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.
    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """

    graph = nx.DiGraph()
    initial_node = {'id': seed, 'level': 0}
    nodes_to_crawl = [initial_node]
    crawled_nodes = set()

    while nodes_to_crawl and len(crawled_nodes) < max_nodes_to_crawl:
        # Pop a node from the nodes to crawl list based on the strategy
        if strategy == "BFS":
            node = nodes_to_crawl.pop(0)
        elif strategy == "DFS":
            node = nodes_to_crawl.pop()
        else:
            raise ValueError("Invalid strategy. Choose either 'BFS' or 'DFS'")

        # Check if the node has already been crawled
        if node['id'] in crawled_nodes:
            continue

        # Add the node to the graph
        graph.add_node(node['id'], level=node['level'])
        crawled_nodes.add(node['id'])

        # Get related artists from Spotify
        results = sp.artist_related_artists(node['id'])
        related_artists = results['artists']

        # Add related artists as nodes and edges in the graph
        for artist in related_artists:
            artist_id = artist['id']
            artist_name = artist['name']
            artist_follower = artist['followers']['total']
            a_genres = artist['genres']
            a_popularity = artist['popularity']

            if artist_id not in crawled_nodes:
                nodes_to_crawl.append({'id': artist_id, 'level': node['level'] + 1, 'name': artist_name, 'followers': artist_follower, 'genres': a_genres, 'popularity': a_popularity})

            genres_str = ", ".join(a_genres)
            graph.add_node(artist_id, name=artist_name, level=node['level'] + 1, followers=artist_follower, genres=genres_str, popularity=a_popularity, label=artist_name)
            graph.add_edge(node['id'], artist_id) #lasciamo gli id perche sono univoci

    #save
    nx.write_graphml(graph, out_filename)
    return graph



'''
The function will receive a list of networkx graphs (where nodes represent artists) and will return a pandas dataframe with the top songs in Spain for all the artists that have been explored in those graphs (each row of the dataframe will represent a song). In addition, the function will save the data from the dataframe in csv format (comma-separated values) to the file indicated by the parameter.
'''

def get_track_data(graphs: list, out_filename: str) -> pd.DataFrame:
    """
    Get track data for each visited artist in the graph.
    :param graphs: a list of graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with track data.
    """
    track_data = []

    for graph in graphs:
        for artist_id in graph.nodes():
            artist_data = graph.nodes[artist_id]

            if 'name' not in artist_data:
                continue

            artist_name = artist_data['name']

            # Get top tracks for the artist in Spain
            results = sp.artist_top_tracks(artist_id, country='ES')
            top_tracks = results['tracks']

            for track in top_tracks:
                track_id = track['id']
                track_name = track['name']
                track_duration = track['duration_ms']
                track_popularity = track['popularity']

                # Get album data for the track
                album_id = track['album']['id']
                album_name = track['album']['name']
                release_date = track['album']['release_date']

                # Get audio features for the track
                audio_features = sp.audio_features(track_id)[0]
                danceability = audio_features['danceability']
                energy = audio_features['energy']
                loudness = audio_features['loudness']
                speechiness = audio_features['speechiness']
                acousticness = audio_features['acousticness']
                instrumentalness = audio_features['instrumentalness']
                liveness = audio_features['liveness']
                valence = audio_features['valence']
                tempo = audio_features['tempo']

                track_data.append({
                    'Artist ID': artist_id,
                    'Artist Name': artist_name,
                    'Track ID': track_id,
                    'Track Name': track_name,
                    'Duration': track_duration,
                    'Popularity': track_popularity,
                    'Album ID': album_id,
                    'Album Name': album_name,
                    'Release Date': release_date,
                    'Danceability': danceability,
                    'Energy': energy,
                    'Loudness': loudness,
                    'Speechiness': speechiness,
                    'Acousticness': acousticness,
                    'Instrumentalness': instrumentalness,
                    'Liveness': liveness,
                    'Valence': valence,
                    'Tempo': tempo
                })

    # Create a DataFrame from the track data
    df = pd.DataFrame(track_data)
    #Save the DataFrame to a CSV file
    df.to_csv(out_filename, index=False)

    return df


#trials
if __name__ == "__main__":

    #1--
    artist_id = search_artist("Ozuna")
    artist_id2 = search_artist("Lunay")
    #print(artist_id)

    #2---
    graph = crawler(artist_id2, max_nodes_to_crawl=5, strategy="BFS", out_filename="//")
    graph = crawler(artist_id, max_nodes_to_crawl=5, strategy="BFS", out_filename="//")
    print(list(graph.nodes(data=True)))

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph, k=0.15) 

    default_label = 'Default Label'
    nx.set_node_attributes(graph, default_label, 'label')
    #nx.draw(graph, pos, with_labels=True, node_size=100, font_size=8, width=0.5, alpha=0.8)
    nx.draw_networkx(graph,pos, with_labels=True, node_color=(.7,.8,.8), font_size=8, width=0.5, alpha=0.8)
    plt.show()

    #3--
    artist_id1=search_artist("Drake")
    graph1 = crawler(artist_id1, max_nodes_to_crawl=5, strategy="BFS", out_filename="//")
    graff= [graph, graph1]
    #graff= [graph]

    track_data = get_track_data(graff, out_filename="//")
    print(track_data)
    


    




    

    
   



    