# This is a sample Python script.

import mat73
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import umap
import plotly.express as px
from sklearn.preprocessing import minmax_scale
import single_unit_analyzer

def split_dataframe(df, chunk_size = 10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        print("on chunk number " +str(i) + " : " + str(round((i/num_chunks), 3)*100) + "%")
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

##bin into 10 ms bins and 1 if there is at least one spike in the bin 0 otherwise
def bin_data(df):
    df_split = split_dataframe(df, chunk_size=1) ##binning here
    return df_split

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

def assign_spike_values_to_bins(binned_data, sigma, mu):
    out_df = None

    threshold = mu + sigma

    for i, chunk in enumerate(binned_data):
        #print(((chunk > 0).any(axis=0)))
        if i == 0:
            out_df = pd.DataFrame(((chunk > threshold).any(axis=0))).transpose() ##threshold here
        else:
            out_df.loc[i] = ((chunk > threshold).any(axis=0))
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

    for i in range(len(vec2)-1):
        if vec2[i] == 1 or vec2[i+1] == 1:
            vec[i] = 1
        else:
            vec[i] = 0

    result = simMI(vec1, vec)
    return result

def create_graph(data):

    data = np.array(data)
    graph = np.zeros((183, 183))

    for i in range(183):
        for j in range(183):
            graph[i][j] = compute_conmi(data[:, i], data[:, j])

    graph[np.isnan(graph)] = 0
    #graph[corr < 0] = 0
    np.fill_diagonal(graph, 0)

    return graph

def compute_graph_centrality(graph):
    deg_centrality = nx.degree_centrality(graph)

    plt.plot(*zip(*sorted(deg_centrality.items())))
    plt.title("Degree Centrality")
    plt.show()

    close_centrality = nx.closeness_centrality(graph)

    plt.plot(*zip(*sorted(close_centrality.items())))
    plt.title("Close Centrality")
    plt.show()

    bet_centrality = nx.betweenness_centrality(graph, normalized=True, endpoints=False)

    plt.plot(*zip(*sorted(bet_centrality.items())))
    plt.title("Between Centrality")
    plt.show()

    pr = nx.pagerank(graph, alpha=0.8)

    plt.plot(*zip(*sorted(pr.items())))
    plt.title("Page Rank")
    plt.show()


def show_graph_with_labels(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    layout = nx.spring_layout(G)

    color_lookup = {k: v for v, k in enumerate(sorted(set(G.nodes())))}
    low, *_, high = sorted(color_lookup.values())
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

    nx.draw(G, layout, nodelist=color_lookup, node_color=[mapper.to_rgba(i) for i in color_lookup.values()])
    nx.draw_networkx_edges(G, pos=layout)
    plt.show()

    '''
    threshold = 0.25
    G.remove_edges_from([(n1, n2) for n1, n2, w in G.edges(data="weight") if abs(w) < threshold])
    nx.draw(G, nodelist=color_lookup, node_color=[mapper.to_rgba(i) for i in color_lookup.values()])
    plt.show()
    '''

    return G

def compute_umap(data):
    neural_data = np.array(data)[:, :]
    print(neural_data)
    print(neural_data)

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


def loop_through_reaches(reaches):
    for reach in reaches:
        #print(reach)
        print('----------------------------------')
        df = pd.DataFrame(reach)
        print(df)
        binned_data = bin_data(df)

        sigma = np.std(df)
        mu = np.mean(df)

        spiked_binned_data = assign_spike_values_to_bins(binned_data, sigma[0], mu)
        graph_adjacency_matrix = create_graph(spiked_binned_data)
        plt.imshow(graph_adjacency_matrix)
        plt.show()

        graph = show_graph_with_labels(graph_adjacency_matrix)

        compute_graph_centrality(graph)
        #compute_umap(np.array(df))

def loop_throgh_non_reaches():
    print()

def subset_reaches(data, masks):
    return data[np.where(masks != 0)[0], :] ##!= gets all reaches

def subset_non_reaches(data, masks):
    return data[np.where(masks == 0)[0], :] ##!= gets all reaches

def subset_individual_reaches(data):
    return data[28648:28662, :]

def subset_reaches_by_frame_start_and_end(data, start_and_end):
    res_list = []
    for datum in start_and_end:
        res2 = data[int(datum[0]):int(datum[1]), :]
        res_list.append(res2)
    return res_list

def main():

    unit_analyzer = single_unit_analyzer.single_unit_analyzer()

    data_dict = mat73.loadmat('/home/macleanlab/Downloads/evaluation_mouse98_0415/evaluated_mouse98_20220415_20230831-172942.mat')
    reach_masks = pd.read_csv('/home/macleanlab/Downloads/20220415_mouse98_allevents_cam1DLC_processed_reachlogical_30Hz.csv')
    reach_begin_end_indices = pd.read_csv('/home/macleanlab/Downloads/20220415_mouse98_allevents_cam1DLC_processed_reachindices_30Hz.csv')

    #print(data_dict)
    neural_data = data_dict['neuron']['C']
    ##print(neural_data.shape)
    df = create_pandas_df_transpose(neural_data)

    #df = pd.DataFrame(subset_reaches(np.array(df), reach_masks))
    #df = pd.DataFrame(subset_individual_reaches(np.array(df)))
    reach_list = subset_reaches_by_frame_start_and_end(df.to_numpy(), reach_begin_end_indices.to_numpy())
    loop_through_reaches(reach_list)

    ##print(reach_begin_end_indices)
    ##print('-----------------------------------------------------------')
    ##print(df)

    ##print(df)
    ##sigma = np.std(df)
    ##mu = np.mean(df)
    ##print("sigma and mu")
    ##print(sigma[0])
    ##print(mu)

    ##print(scipy.ndimage.gaussian_filter1d(df, 1))

    ##binned_data = bin_data(df)
    ##spiked_binned_data = assign_spike_values_to_bins(binned_data, sigma[0], mu)
    ##graph_adjacency_matrix = create_graph(spiked_binned_data)
    ##plt.imshow(graph_adjacency_matrix)
    ##plt.show()

    ##graph = show_graph_with_labels(graph_adjacency_matrix)

    '''
    unit_analyzer.set_graph(graph)
    unit_analyzer.compute_top_central_units()
    print(graph)
    '''
    ##compute_graph_centrality(graph)
    ##compute_umap(np.array(df))


if __name__ == '__main__':
    main()

