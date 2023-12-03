import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Lab_AGX_202223_S3_skeleton import get_degree_distribution #analysis file= S3
from Lab_AGX_202223_S2_skeleton import compute_mean_audio_features #preproc file= S2


#DA S3 get degree distribution
def plot_degree_distribution(degree_dict: dict, normalized: bool = False, loglog: bool = False) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    gradi = list(degree_dict.keys())
    conteggi = list(degree_dict.values())

    if normalized:
        totale_conteggi = sum(conteggi)
        probabilità = [conteggio / totale_conteggi for conteggio in conteggi]
        conteggi = probabilità

    if loglog:
        plt.loglog(gradi, conteggi, 'bo-')
        plt.xlabel('Grado')
        plt.ylabel('Distribuzione dei Gradi (Scala Logaritmica)')
    else:
        plt.plot(gradi, conteggi, 'bo-')
        plt.xlabel('Grado')
        plt.ylabel('Distribuzione dei Gradi')

    plt.title('Distribuzione dei Gradi')
    plt.show()


#DA S2 compute mean audio features
def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """
    Plot a (single) figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    # Filtra il dataframe per ottenere le righe relative agli artisti specificati
    artist1_data = artists_audio_feat[artists_audio_feat['Artist ID'] == artist1_id]
    artist2_data = artists_audio_feat[artists_audio_feat['Artist ID'] == artist2_id]

    # Definisci gli attributi desiderati
    features = ['Danceability', 'Energy', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence']

    # Prendi i valori medi delle caratteristiche audio per i due artisti
    artist1_mean = artist1_data[features].values.flatten()
    artist2_mean = artist2_data[features].values.flatten()

    # Crea un grafico a barre per confrontare le caratteristiche audio medie
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = list(range(len(features)))  # Converti in una lista

    # Disegna le barre per i due artisti
    rects1 = ax.bar(index, artist1_mean, bar_width, label=artist1_id)
    rects2 = ax.bar([i + bar_width for i in index], artist2_mean, bar_width, label=artist2_id)  # Aggiungi bar_width al valore

    # Aggiungi le etichette dell'asse x
    ax.set_xlabel('Audio Features')
    ax.set_ylabel('Mean Value')
    ax.set_xticks([i + bar_width / 2 for i in index])  # Aggiungi bar_width / 2 al valore
    ax.set_xticklabels(features, rotation=45)

    # Aggiungi una legenda
    ax.legend()
    plt.show()


#DA S2 compute mean audio features
def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    # Calculate the similarity matrix
    similarity_matrix = artist_audio_features_df.corr(method=similarity)
    
    # Create the heatmap using seaborn
    fig, ax = plt.subplots()
    sns.heatmap(similarity_matrix, cmap="coolwarm", annot=True, fmt=".2f")

    # Add a title to the plot
    plt.title("Similarity Heatmap")

    # Save the plot if the output filename is specified
    if out_filename:
        plt.savefig(out_filename)

    # Show the plot
    plt.show()





if __name__ == "__main__":
    #1
    #qua butto la funzione per ottenere il dizionario da s3
    #poi applico la mia funzione nuova
    file1 = "//"
    g1 = nx.read_graphml(file1)
    distribution = get_degree_distribution(g1)

    plot_degree_distribution(distribution, normalized=True, loglog=False)
    
    #2
    tracks_df = pd.read_csv("//")
    # Call the compute_mean_audio_features function
    result_df = compute_mean_audio_features(tracks_df)
    #print(result_df)

    print(result_df.columns)
    artist1_id = '7iK8PXO48WeuP03g8YR51W'
    artist2_id = '0EeQBlQJFiAfJeVN2vT9s0'

    plot_audio_features(result_df, artist1_id, artist2_id)

    #3
    similarity_measure = "pearson"  # Replace with your desired similarity measure
    plot_similarity_heatmap(result_df, similarity_measure, out_filename="//")
