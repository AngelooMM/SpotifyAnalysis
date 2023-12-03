import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

'''
The function will receive a directed graph as input and return an undirected graph. The resulting undirected graph will have as edges those edges from the original graph that existed in both directions. In other words, the edge e = (vi,vj) will exist in the new graph if and only if both edges e = (vi,vj) and e′ = (vj,vi) existed in the input graph. The nodes of the new graph will be defined by the edges contained in the new graph (only nodes that have incident edges in the new graph should appear). The function retrieve bidirectional edges will return the resulting graph and, additionally, it will save it to disk in a file with the specified name as a parameter (in graphml format).
'''
def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    #undirected_graph.add_nodes_from(g)
    undirected_graph = nx.Graph()

    for edge in g.edges():
        if g.has_edge(edge[1], edge[0]):
            undirected_graph.add_edge(edge[0], edge[1])

    nx.write_graphml(undirected_graph, out_filename)
    return undirected_graph
    

''' 
The function will receive an undirected input graph and generate a graph from which all nodes with a degree less than min degree have been removed. The function will have three input parameters: an undirected networkx graph, a variable with the minimum degree value, and the output file name where the generated graph will be saved.
This removal of nodes with a degree less than min degree should be done in a single pass. In other words, at the end of the traversal of all nodes, there may still be nodes with a degree less than min degree. In this case, we will keep these nodes. After this process, you should remove the zero-degree nodes from the resulting graph.
'''
def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    pruned_graph = g.copy() # Crea una copia del grafo originale
    
    # Trova i nodi con grado inferiore al grado minimo
    nodes_to_remove = [node for node, degree in pruned_graph.degree() if degree < min_degree]
    
    # Rimuovi i nodi con grado inferiore al grado minimo
    pruned_graph.remove_nodes_from(nodes_to_remove)
    
    # Rimuovi i nodi di grado zero
    zero_degree_nodes = [node for node, degree in pruned_graph.degree() if degree == 0]
    pruned_graph.remove_nodes_from(zero_degree_nodes)
    
    # Salva il grafo ridotto su file
    nx.write_graphml(pruned_graph, out_filename)
    return pruned_graph


''' 
The function will receive an undirected input graph with weighted edges and gen- erate a graph from which all edges with a weight less than the value specified as a parameter will be removed. This weight can be specified in two different ways: either directly with the threshold value (parameter min weight) or with the per- centile value (parameter min percentile). The function should raise an exception if neither of the two parameters is specified or if both parameters are specified (i.e., the function call should only specify one of the two parameters).
After the edge removal process, you should remove the zero-degree nodes from the resulting graph.
In addition to returning the resulting graph, the function will also save it to disk in the file specified by the parameter name (in graphml format).
'''
def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    if (min_weight is None and min_percentile is None) or (min_weight is not None and min_percentile is not None):
        raise ValueError("Specify only one of the two parameters: min_weight or min_percentile.")

    pruned_graph = g.copy()  # Create a copy of the original graph

    if min_weight is not None:
        # Remove edges with weight lower than the minimum threshold
        edges_to_remove = [(u, v) for u, v, weight in pruned_graph.edges(data="weight") if weight is not None and weight < min_weight]
        pruned_graph.remove_edges_from(edges_to_remove)

    if min_percentile is not None:
        # Calculate the percentile threshold
        weight_values = [weight for u, v, weight in pruned_graph.edges(data="weight") if weight is not None]
        threshold = nx.utils.percentile(weight_values, min_percentile)

        # Remove edges with weight lower than the percentile threshold
        edges_to_remove = [(u, v) for u, v, weight in pruned_graph.edges(data="weight") if weight is not None and weight < threshold]
        pruned_graph.remove_edges_from(edges_to_remove)

    # Remove zero-degree nodes
    zero_degree_nodes = [node for node, degree in pruned_graph.degree() if degree == 0]
    pruned_graph.remove_nodes_from(zero_degree_nodes)

    nx.write_graphml(pruned_graph, out_filename)
    return pruned_graph


''' 
The function will receive a dataframe of songs (the result of the get track data function implemented in session 1) and will return another dataframe with the average audio characteristics of each artist. In other words, the resulting dataframe will have one row for each artist, which will contain both the identification data of that artist (at least, the identifier and name) and the average values of each audio characteristic (danceability, energy, loudness, ...) of all the songs by that artist.
'''
def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    artist_df = tracks_df.groupby(['Artist ID', 'Artist Name']).mean().reset_index()
    return artist_df



def calculate_similarity(artist1, artist2, similarity):
    if similarity == "cosine":
        # Calcola la similarità coseno tra gli artisti
        features1 = artist1.drop(['Artist ID', 'Artist Name']).values.reshape(1, -1)
        features2 = artist2.drop(['Artist ID', 'Artist Name']).values.reshape(1, -1)
        similarity_score = cosine_similarity(features1, features2)[0, 0]
    elif similarity == "euclidean":
        # Calcola la similarità euclidea tra gli artisti
        features1 = artist1.drop(['Artist ID', 'Artist Name']).values.reshape(1, -1)
        features2 = artist2.drop(['Artist ID', 'Artist Name']).values.reshape(1, -1)
        similarity_score = 1 / (1 + euclidean_distances(features1, features2)[0, 0])
    else:
        raise ValueError("Invalid similarity metric.")
    return similarity_score



def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.
    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    graph = nx.Graph()

    # Iterate over pairs of artists
    for i, artist1 in artist_audio_features_df.iterrows():
        for j, artist2 in artist_audio_features_df.iterrows():
            # Skip self-pairs
            if i != j:
                # Calculate similarity between artists
                similarity_score = calculate_similarity(artist1, artist2, similarity)

                # Add edge with similarity as weight
                graph.add_edge(artist1['Artist ID'], artist2['Artist ID'], weight=similarity_score)

    # Save the graph to file
    if out_filename is not None:
        nx.write_graphml(graph, out_filename)

    return graph



if __name__ == "__main__":
    #normale di lab1
    #undirected
    #pruned
    #pruned2
    #dt
    
    #1
    graph = nx.read_graphml("//") #poi salvalo directed_graph
    
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph, k=0.15) 
    nx.draw_networkx(graph,pos, with_labels=True, node_color=(.7,.8,.8), font_size=8, width=0.5, alpha=0.8)
    plt.show()
    
    undirected_graph = retrieve_bidirectional_edges(graph, "//")
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(undirected_graph, k=0.15) 

    nx.draw_networkx(undirected_graph,pos, with_labels=True, node_color=(.7,.8,.8), font_size=8, width=0.5, alpha=0.8)
    plt.show()

    #2
    pruned_graph = prune_low_degree_nodes(undirected_graph, min_degree=3, out_filename="//")
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(pruned_graph, k=0.15)
    nx.draw_networkx(pruned_graph, pos, with_labels=True, node_color=(.7, .8, .8), font_size=8, width=0.5, alpha=0.8)
    plt.show()

    #3
    pruned_graph = prune_low_weight_edges(pruned_graph, min_weight=0.5, out_filename="//")
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(pruned_graph, k=0.15)
    nx.draw_networkx(pruned_graph, pos, with_labels=True, node_color=(.7, .8, .8), font_size=8, width=0.5, alpha=0.8)
    plt.show()

    #4
    # Load the tracks dataframe from CSV
    tracks_df = pd.read_csv("//")
    #print(tracks_df)
    # Call the compute_mean_audio_features function
    result_df = compute_mean_audio_features(tracks_df)
    print(result_df)

    #5
    #Call the create_similarity_graph function
    print("Ultimoooooo")

    # Call the create_similarity_graph function
    similarity_graph = create_similarity_graph(result_df, similarity="cosine", out_filename="//")
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(similarity_graph, k=0.15)
    nx.draw_networkx(similarity_graph, pos, with_labels=True, node_color=(.7, .8, .8), font_size=8, width=0.5, alpha=0.8)
    plt.show()

    

