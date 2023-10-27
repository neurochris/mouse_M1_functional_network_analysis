# This is a sample Python script.

import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import umap
import plotly.express as px


def split_dataframe(df, chunk_size = 10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        print("on chunk number " +str(i) + " : " + str(round((i/num_chunks), 3)*100) + "%")
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

##bin into 10 ms bins and 1 if there is at least one spike in the bin 0 otherwise
def bin_data(df):
    df_split = split_dataframe(df, chunk_size=3)
    return df_split

def plot_data(data):
    for idx in range(1, len(data)):
        if idx%1000 == 0:
            neural_data = np.array(data.iloc[idx:idx+1])[0]
            print(neural_data)
            spike_times = [i for i, x in enumerate(neural_data) if x == 1]
            fig, ax = plt.subplots()
            ax.vlines(spike_times, 0, 1)
            plt.show()

def create_pandas_df_transpose(data):
    #print(data)
    df = pd.DataFrame(data)
    #print(df.T)
    return df.T

def create_pandas_df(data):
    #print(data)
    df = pd.DataFrame(data)
    #print(df.T)
    return df

def assign_spike_values_to_bins(binned_data):
    out_df = None
    for i, chunk in enumerate(binned_data):
        #print(((chunk > 0).any(axis=0)))
        if i == 0:
            out_df = pd.DataFrame(((chunk > 0).any(axis=0))).transpose()
        else:
            out_df.loc[i] = ((chunk > 0).any(axis=0))
    print(out_df.astype(int))
    return out_df.astype(int)

def simMI(vec1, vec2):
    dist_vec1 = np.histogram(vec1, bins=2)[0]
    dist_vec2 = np.histogram(vec2, bins=2)[0]
    dist_2d = np.histogram2d(vec1, vec2, bins=2)[0]

    print(dist_vec1)
    print(dist_vec2)
    print(dist_2d)

    p_vec1_0 = dist_vec1[0]/vec1.size
    p_vec1_1 = dist_vec1[1]/vec1.size
    p_vec2_0 = dist_vec2[0]/vec2.size
    p_vec2_1 = dist_vec2[1]/vec2.size

    p_00 = dist_2d[0][0]/vec1.size
    p_01 = dist_2d[0][1]/vec1.size
    p_10 = dist_2d[1][0]/vec1.size
    p_11 = dist_2d[1][1]/vec1.size

    print(p_vec1_0)
    print(p_vec1_1)
    print(p_vec2_0)
    print(p_vec2_1)

    print(p_00)
    print(p_01)
    print(p_10)
    print(p_11)

    result = p_00*np.log2(p_00/(p_vec1_0*p_vec2_0)) + p_10*np.log2(p_10/(p_vec1_1*p_vec2_0)) + p_01*np.log2(p_01/(p_vec1_0*p_vec2_1)) + p_11*np.log2(p_11/(p_vec1_1*p_vec2_1))

    return result


def compute_conmi(vec1, vec2):

    vec = vec2

    for i in range(len(vec1)-1):
        if vec2[i] == 1 or vec2[i+1] == 1:
            vec[i] = 1
        else:
            vec[i] = 0

    result = simMI(vec1, vec)
    return result

def create_graph(data):

    data = np.array(data)
    graph = np.zeros((60, 60))

    for i in range(60):
        for j in range(60):
            graph[i][j] = compute_conmi(data[i, :], data[j, :])

    graph[np.isnan(graph)] = 0
    #graph[corr < 0] = 0
    np.fill_diagonal(graph, 0)

    return graph

def compute_graph_centrality(graph):
    deg_centrality = nx.degree_centrality(graph)
    print(deg_centrality)

    plt.plot(*zip(*sorted(deg_centrality.items())))
    plt.title("Degree Centrality")
    plt.show()

    close_centrality = nx.closeness_centrality(graph)
    print(close_centrality)

    plt.plot(*zip(*sorted(close_centrality.items())))
    plt.title("Close Centrality")
    plt.show()

    bet_centrality = nx.betweenness_centrality(graph, normalized=True, endpoints=False)
    print(bet_centrality)

    plt.plot(*zip(*sorted(bet_centrality.items())))
    plt.title("Between Centrality")
    plt.show()

    pr = nx.pagerank(graph, alpha=0.8)
    print(pr)

    plt.plot(*zip(*sorted(pr.items())))
    plt.title("Page Rank")
    plt.show()


def show_graph_with_labels(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    layout = nx.spring_layout(G)
    nx.draw(G, layout)
    nx.draw_networkx_edges(G, pos=layout)
    plt.show()
    return G

def compute_umap(data):
    neural_data = np.array(data)[1:2000, :]
    print(neural_data.shape)

    umap_2d = umap.UMAP(n_components=2, init='pca', random_state=0)
    umap_3d = umap.UMAP(n_components=3, init='pca', random_state=0)

    proj_2d = umap_2d.fit_transform(neural_data)
    proj_3d = umap_3d.fit_transform(neural_data)

    fig_2d = px.scatter(
        proj_2d, x=0, y=1,
    )
    fig_3d = px.scatter_3d(
        proj_3d, x=0, y=1, z=2,
    )
    fig_3d.update_traces(marker_size=5)

    fig_2d.show()
    fig_3d.show()

def main():
    data_dict = mat73.loadmat('/media/macleanlab/DATA/IMAGING_DATA/caimanoutput_20231019-172438/evaluation/evaluated_IMAGING_DATA_20231020-164721.mat')
    #print(data_dict)
    neural_data = data_dict['neuron']['C']
    print(neural_data.shape)
    df = create_pandas_df_transpose(neural_data)
    binned_data = bin_data(df)
    spiked_binned_data = assign_spike_values_to_bins(binned_data)
    #plot_data(spiked_binned_data)
    graph_adjacency_matrix = create_graph(spiked_binned_data)
    print(graph_adjacency_matrix)
    graph = show_graph_with_labels(graph_adjacency_matrix)
    compute_graph_centrality(graph)
    compute_umap(spiked_binned_data)

if __name__ == '__main__':
    main()

