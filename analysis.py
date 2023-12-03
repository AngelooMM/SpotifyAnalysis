import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import community



''' 
The function will receive an undertermined number of network graphs as input. The result of the function will be an integer counting the number of nodes that appear in all of these graphs.
'''
def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    if len(arg) == 0:
        return 0

    common_nodes = set(arg[0].nodes())

    for graph in arg[1:]:
        common_nodes = common_nodes.intersection(set(graph.nodes()))

    return len(common_nodes)


''' 
The function will receive a graph and will return its degree distribution as a dic- tionary (keys will be the degrees and values will be the number of nodes with that specific degree).
'''
def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    degree_distribution = {}

    # Calculate the degree for each node in the graph
    degrees = [g.degree(node) for node in g.nodes()]

    # Count the occurrences of each degree
    for degree in degrees:
        if degree in degree_distribution:
            degree_distribution[degree] += 1
        else:
            degree_distribution[degree] = 1

    return degree_distribution




''' 
The function will receive a graph, a centrality metric name, and the number of nodes to return. The function will return the k most central nodes according to the specified metric. The function should allow calculating, at least, degree centrality, betweenness centrality, closeness centrality, and eigenvector centrality.
'''
def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes with the specified centrality.
    """
    if metric == 'degree':
        centrality_scores = nx.degree_centrality(g)
    elif metric == 'betweenness':
        centrality_scores = nx.betweenness_centrality(g)
    elif metric == 'closeness':
        centrality_scores = nx.closeness_centrality(g)
    elif metric == 'eigenvector':
        centrality_scores = nx.eigenvector_centrality(g)
    else:
        raise ValueError("Invalid centrality metric. Please choose from 'degree', 'betweenness', 'closeness', or 'eigenvector'.")

    # Sort nodes based on centrality scores
    sorted_nodes = sorted(centrality_scores, key=centrality_scores.get, reverse=True)

    # Return the top num_nodes nodes
    return sorted_nodes[:num_nodes]



'''
The function will search for cliques in a graph. The function will receive a graph as input, as well as the parameter min size clique, and will return two lists. The first list will contain each of the cliques with a size greater than or equal to min size clique. The second list will include all the different nodes that are part of any of these cliques.'''
def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    cliques = [clique for clique in nx.find_cliques_recursive(g) if len(clique) >= min_size_clique]
    nodes_in_cliques = list(set(node for clique in cliques for node in clique))

    return cliques, nodes_in_cliques

''' 
The function will receive a graph as input and will return a list of lists with the nodes grouped into communities, along with the modularity of the partitioning. The function can receive additional parameters that allow configuring the community detection (e.g., the specific algorithm to implement or the randomness seed if it is not deterministic).
'''
def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    if method == 'girvan-newman':
        communities_generator = nx.algorithms.community.girvan_newman(g)
        communities = next(communities_generator)
    elif method == 'louvain':
        partition = community.best_partition(g)
        communities = [list(nodes) for nodes in partition.values()]
    else:
        raise ValueError("Metodo di rilevamento delle comunità non valido. Scegli 'girvan-newman' o 'louvain'.")

    modularity = nx.algorithms.community.modularity(g, communities)

    return communities, modularity



if __name__ == '__main__':
    #1
    # Definisci il percorso del file del grafo
    file1 = "//"
    file2 = "//"

    # Leggi il grafo dal file
    g1 = nx.read_graphml(file1)
    g2 = nx.read_graphml(file2)

    common_nodes_count = num_common_nodes(g1,g2)
    print("Number of common nodes:", common_nodes_count)

    #2
    #Call the function
    distribution = get_degree_distribution(g1) #faccio sempre del grafo 1
    print("Esercizio 2: ",distribution)

    #3
    # Get the 3 most central nodes based on degree centrality
    k_most_central_nodes = get_k_most_central(g1, 'degree', 5)
    print("Nodi centrali",k_most_central_nodes)

    #4
    # Find cliques with a minimum size of 3
    cliques, nodes_in_cliques = find_cliques(g1, 3)
    print("Cliques:"+"\n", cliques)
    print("Nodes in Cliques:", nodes_in_cliques)

    #5
    file3 = "//"
    # Leggi il grafo dal file
    g3 = nx.read_graphml(file3)
    # Detect communities using Girvan-Newman method
    communities, modularity = detect_communities(g3, 'girvan-newman')
    print("Communities (Girvan-Newman):")
    for i, community in enumerate(communities):
        print(f"Community {i + 1}: {community}")
    print("Modularity:", modularity)