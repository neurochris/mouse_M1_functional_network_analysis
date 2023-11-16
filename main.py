# This is a sample Python script.

import mat73
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import networkx as nx
import umap
import plotly.express as px
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import single_unit_analyzer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import zscore
import networkx.algorithms.isomorphism as iso

def zify_scipy(d):
    keys, vals = zip(*d.items())
    return dict(zip(keys, zscore(vals, ddof=1)))

c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
v = [0, .15, .4, .5, 0.6, .9, 1.]
l = list(zip(v, c))
cmap = LinearSegmentedColormap.from_list('rg', l, N=256)


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
    return out_df.astype(int)

def simMI(vec1, vec2):
    dist_vec1 = np.histogram(vec1, bins=2)[0]
    dist_vec2 = np.histogram(vec2, bins=2)[0]
    dist_2d = np.histogram2d(vec1, vec2, bins=2)[0]

    p_vec1_0 = dist_vec1[0]/vec1.size
    p_vec1_1 = dist_vec1[1]/vec1.size
    p_vec2_0 = dist_vec2[0]/vec2.size
    p_vec2_1 = dist_vec2[1]/vec2.size

    p_00 = dist_2d[0][0]/vec1.size
    p_01 = dist_2d[0][1]/vec1.size
    p_10 = dist_2d[1][0]/vec1.size
    p_11 = dist_2d[1][1]/vec1.size

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

def compute_graph_centrality(graph, i):
    deg_centrality = nx.degree_centrality(graph)

    plt.plot(*zip(*sorted(deg_centrality.items())))
    plt.title("Degree Centrality "+str(i))
    plt.show()

    close_centrality = nx.closeness_centrality(graph)


    plt.plot(*zip(*sorted(close_centrality.items())))
    plt.title("Close Centrality" +str(i))
    plt.show()

    bet_centrality = nx.betweenness_centrality(graph, normalized=True, endpoints=False)

    plt.plot(*zip(*sorted(bet_centrality.items())))
    plt.title("Between Centrality"+str(i))
    plt.show()

    #pr = nx.pagerank(graph, alpha=0.8)

    #plt.plot(*zip(*sorted(pr.items())))
    #plt.title("Page Rank")
    #plt.show()

    return bet_centrality


def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    #plt.hist(degrees)
    #plt.show()
    n, bins, patches = plt.hist(degrees, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.8, alpha=0.9)


    n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
    for i in range(len(patches)):
        patches[i].set_facecolor(cmap(n[i] / max(n)))  # Make one bin stand out
    '''
    patches[47].set_fc('red')  # Set color
    patches[47].set_alpha(1)  # Set opacity# Add annotation
    plt.annotate('Important Bar!', xy=(0.57, 175), xytext=(2, 130), fontsize=15,
                 arrowprops={'width': 0.4, 'headwidth': 7,
                             'color': '#333333'})  # Add title and labels with custom font sizes
    '''
    plt.title('Reach Degree Distribution', fontsize=12)
    plt.xlabel('Bins', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.show()

def plot_weight_dist(G):
    weights = [G.weight(n) for n in G.nodes()]
    plt.hist(weights)
    plt.show()

def show_graph_with_labels(adjacency_matrix, counter=0):

    upper_quartile = False

    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    layout = nx.spring_layout(G)


    color_lookup = {k: v for v, k in enumerate(sorted(set(G.nodes())))}
    low, *_, high = sorted(color_lookup.values())
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

    nx.draw(G, layout, nodelist=color_lookup, node_color=[mapper.to_rgba(i) for i in color_lookup.values()])

    nx.draw_networkx_edges(G, pos=layout)

    prefix = '/home/macleanlab/Desktop/chris_data_out/'
    save_path = prefix + 'graphs/' + str(counter) + '.png'
    plt.title("reaching stage " + str(counter))

    plt.show()

    c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
    v = [0, .15, .4, .5, 0.6, .9, 1.]
    l = list(zip(v, c))
    cmap = LinearSegmentedColormap.from_list('rg', l, N=256)

    if upper_quartile == True:
        threshold = np.percentile(adjacency_matrix.flatten(), 75)




        ##75 for non reach non graph figures - figs on right
        ##try 25-50 next





        G.remove_edges_from([(n1, n2) for n1, n2, w in G.edges(data="weight") if abs(w) < threshold])
        deg_centrality = nx.degree_centrality(G)
        #nx.draw(G, layout, nodelist=color_lookup, node_color=[mapper.to_rgba(i) for i in color_lookup.values()])
        #plt.savefig(fname=save_path, format='png')
        #plt.show()
        print(deg_centrality)
        cent = np.fromiter(deg_centrality.values(), float)
        sizes = cent / np.max(cent) * 350
        normalize = mcolors.Normalize(vmin=cent.min(), vmax=cent.max())
        colormap = cmap

        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(cent)
        fig, ax = plt.subplots()
        plt.title("Reach Graph (95th Percentile)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        plt.colorbar(scalarmappaple, cax=cax)
        nx.draw(G, layout, ax=ax, node_size=sizes, node_color=sizes, cmap=colormap)
        plt.show()


    return G

def compute_umap(data, counter=0):
    neural_data = np.array(data)[:, :]
    print(neural_data)
    print(neural_data)

    umap_2d = umap.UMAP(n_components=2, init='pca', random_state=0)
    umap_3d = umap.UMAP(n_components=3, init='pca', random_state=0)

    try:
        proj_2d = umap_2d.fit_transform(neural_data)
        proj_3d = umap_3d.fit_transform(neural_data)

        fig_2d = px.scatter(
            proj_2d, x=0, y=1,
        )
        fig_3d = px.scatter_3d(
            proj_3d, x=0, y=1, z=2,
        )
        fig_3d.update_traces(marker_size=5)

        prefix = '/home/macleanlab/Desktop/chris_data_out/'
        filename_2d = prefix + '2d/' + str(counter) + '.png'
        filename_3d = prefix + '3d/' + str(counter) + '.png'
        fig_2d.write_image(file=filename_2d, format='png')
        fig_3d.write_image(file=filename_3d, format='png')
    except Exception as e:
        print(e)
        pass

def loop_through_reaches(reaches):
    dc_df = None
    init = True
    for idx, reach in enumerate(reaches):
        if(reach.shape[0] > 6):
            for i in range(2):
                mid = int(reach.shape[0]/2)

                if i == 0:
                    half_reach = reach[0:mid]
                elif i == 1:
                    half_reach = reach[mid:reach.shape[0]]

                #print(reach)
                print('----------------------------------')
                df = pd.DataFrame(half_reach)
                print(df)
                binned_data = bin_data(df)

                sigma = np.std(df)
                mu = np.mean(df)

                spiked_binned_data = assign_spike_values_to_bins(binned_data, sigma[0], mu)
                graph_adjacency_matrix = create_graph(spiked_binned_data)
                #background_graph = background(graph_adjacency_matrix)
                #print('computing res')
                #residual_graph = residual(background_graph, graph_adjacency_matrix)
                #print('computing res norm graph')
                #normed_graph = normed_residual(residual_graph)
                plt.imshow(graph_adjacency_matrix)
                plt.show()
                print('showing graph')
                graph = show_graph_with_labels(graph_adjacency_matrix, i)
                print('computing centrality')
                dc = compute_graph_centrality(graph, i)
                if all(value == 0 for value in dc.values()):
                    print('adding centrality to dataframe')
                    if init:
                        init = False
                        dc_df = pd.DataFrame.from_dict(dc, orient='index')
                        dc_df = dc_df.rename(columns={0: 'reach 0'})
                    else:
                        dc_df["reach" + str(idx)] = pd.Series(dc)
                    print("dc: ")
                    print(dc)
                    print('-------------------------------')
                    print(dc_df)
                    print('-------------------------------')
                    dc_df.to_csv('/home/macleanlab/Desktop/chris_data_out/centrality_csv/'+str(idx)+'.csv')
                    compute_umap(np.array(df), idx)
                    pca = PCA(n_components=2)

                    principalComponents = pca.fit_transform(graph_adjacency_matrix)
                    print(principalComponents)
                    print(principalComponents.shape)
                    print(pca.explained_variance_ratio_)
                    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
                    plt.title("Stage: " + str(i))
                    plt.show()
                    plot_degree_dist(graph)

def loop_throgh_non_reaches():
    print()

def subset_reaches(data, masks):
    return data[np.where(masks != 0)[0], :] ##!= gets all reaches

def subset_non_reaches(data, masks):
    return data[np.where(masks == 0)[0], :] ##!= gets all reaches



def subset_reaches_by_frame_start_and_end(data, start_and_end):
    res_list = []
    for datum in start_and_end:
        res2 = data[int(datum[0]):int(datum[1]), :]
        res_list.append(res2)
    return res_list

def background(reexpress_graph):
    """Return the background graph from a reexpress graph."""
    background = np.copy(reexpress_graph)
    neurons = np.arange(0, np.shape(reexpress_graph)[0])

    for pre in neurons:
        for post in neurons:
            if pre != post:
                mean1 = np.mean(reexpress_graph[pre, neurons[neurons != post]])
                mean2 = np.mean(reexpress_graph[neurons[neurons != pre], post])
                background[pre, post] = mean1 * mean2

    return background

def residual(background_graph, graph):
    """Calculate the residual."""
    residual = np.copy(graph)
    neurons = np.shape(graph)[0]
    lm = LinearRegression()
    # b,m = linreg(background_graph[:],graph[:])
    model = lm.fit(background_graph.reshape(-1, 1), graph.reshape(-1))
    b = model.intercept_
    m = model.coef_
    residual = graph - (m * background_graph + b)
    return residual



def normed_residual(graph):
    """Calculate the normalized residual."""
    norm_residual = np.copy(graph)
    neurons = np.arange(0, np.shape(graph)[0])

    for pre in neurons:
        for post in neurons:
            if pre != post:
                norm_residual[pre, post] = np.std(graph[pre, neurons[neurons != post]]
                ) * np.std(graph[neurons[neurons != pre], post])

    cutoff = np.median(norm_residual)
    neurons = np.shape(graph)[0]
    norm_residual = 1/(np.sqrt(np.maximum(norm_residual, np.ones([neurons, neurons]) * cutoff)))
    return norm_residual * graph

def find_subgraph(Gg):
    # make an undirected copy of the digraph
    UG = Gg.to_undirected()

    # extract subgraphs
    A=list(UG.subgraph(c) for c in nx.connected_components(UG))
    counter = 0
    for sg in A:
        plt.figure()
        G = sg
        layout = nx.spring_layout(G)

        color_lookup = {k: v for v, k in enumerate(sorted(set(G.nodes())))}
        low, *_, high = sorted(color_lookup.values())
        norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

        nx.draw(G, layout, nodelist=color_lookup, node_color=[mapper.to_rgba(i) for i in color_lookup.values()])

        nx.draw_networkx_edges(G, pos=layout)

        prefix = '/home/macleanlab/Desktop/chris_data_out/'
        save_path = prefix + 'graphs/' + str(counter) + '.png'
        counter = counter + 1
        # plt.savefig(fname=save_path, format='png')
        plt.show()

def analyze_subgraphs(G):
    ug_sub = G.to_undirected()

    #LSubG = nx.minimum_spanning_tree(ug_sub)
    #print(LSubG)
    S = nx.tutte_polynomial(ug_sub)
    print(S)

def subset_individual_reaches(data):
    return data[35446:35456, :]

def main():

    load = 2

    #unit_analyzer = single_unit_analyzer.single_unit_analyzer()

    data_dict = mat73.loadmat('/home/macleanlab/Downloads/evaluation_mouse98_0415/evaluated_mouse98_20220415_20230831-172942.mat')
    reach_masks = pd.read_csv('/home/macleanlab/Downloads/20220415_mouse98_allevents_cam1DLC_processed_reachlogical_30Hz.csv')
    reach_begin_end_indices = pd.read_csv('/home/macleanlab/Downloads/20220415_mouse98_allevents_cam1DLC_processed_reachindices_30Hz.csv')
    # print(data_dict)
    neural_data = data_dict['neuron']['C']
    ##print(neural_data.shape)
    df = create_pandas_df_transpose(neural_data)

    if load == 0:

        binned_data = bin_data(df)

        sigma = np.std(df)
        mu = np.mean(df)

        spiked_binned_data = assign_spike_values_to_bins(binned_data, sigma[0], mu)
        graph_adjacency_matrix = create_graph(spiked_binned_data)

        background_graph = background(graph_adjacency_matrix)
        residual_graph = residual(background_graph, graph_adjacency_matrix)
        normed_graph = normed_residual(residual_graph)
        plt.imshow(normed_graph)
        plt.show()

        np.save("/home/macleanlab/Desktop/chris_data_out/numpy/normed_graph.npy", normed_graph)
    elif load == 1:
        normed_graph = np.load("/home/macleanlab/Desktop/chris_data_out/numpy/normed_graph.npy")
        print('normed graph loaded from numpy array!')
        #plt.imshow(normed_graph,cmap='inferno')
        #plt.title("Full Session Weight Distribution")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        axp = ax.imshow(normed_graph, cmap='inferno')
        cb = plt.colorbar(axp, ax=[ax], location='right')
        plt.title("Full Session Weight Distribution")
        plt.show()


    #df = pd.DataFrame(subset_reaches(np.array(df), reach_masks))
    #df = pd.DataFrame(subset_individual_reaches(np.array(df)))
    #binned_data = bin_data(df)

    #sigma = np.std(df)
    #mu = np.mean(df)

    #spiked_binned_data = assign_spike_values_to_bins(binned_data, sigma[0], mu)
    #graph_adjacency_matrix = create_graph(spiked_binned_data)

    #background_graph = background(graph_adjacency_matrix)
    #residual_graph = residual(background_graph, graph_adjacency_matrix)
    #normed_graph = normed_residual(residual_graph)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #axp = ax.imshow(normed_graph, cmap='inferno')
    #cb = plt.colorbar(axp, ax=[ax], location='right')
    #plt.title("Reach Weight Distribution")
    #plt.show()

    reach_list = subset_reaches_by_frame_start_and_end(df.to_numpy(), reach_begin_end_indices.to_numpy())
    print(reach_list)
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

    #graph = show_graph_with_labels(normed_graph)
    #find_subgraph(graph)
    '''
    unit_analyzer.set_graph(graph)
    unit_analyzer.compute_top_central_units()
    print(graph)


    deg = compute_graph_centrality(graph)
    compute_umap(df.to_numpy())

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(df.to_numpy())
    print(principalComponents)
    print(principalComponents.shape)
    print(pca.explained_variance_ratio_)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.show()
    plot_degree_dist(graph)



    deg_z_scores = zify_scipy(deg)
    print(deg_z_scores)
    deg_z_scores_sorted = dict(sorted(deg_z_scores.items(), key=lambda x: x[1]))

    neuron_idx = list(deg_z_scores_sorted.keys())  # list() needed for python 3.x
    deg_z_scores_to_plot = list(deg_z_scores_sorted.values())
    print(neuron_idx)
    print(deg_z_scores)
    # plotting a line plot after changing it's width and height
    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(7)

    new = []
    for j in neuron_idx:
        new.append(str(j))

    plt.plot(list(deg_z_scores.values()), list(deg_z_scores.items()), 'g-')
    plt.xlabel("Standardized Z-Score")
    plt.ylabel("Neuron Index")
    plt.title("Reach Betweenness Centrality")
    plt.show()


    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(7)


    plt.plot(deg_z_scores_to_plot[153:183], new[153:183], 'g-')
    plt.xlabel("Standardized Z-Score")
    plt.ylabel("Neuron Index")
    plt.title("Reach Betweenness Centrality Top N")
    plt.yticks(fontsize=10)
    plt.show()


    #analyze_subgraphs(graph)

    '''

if __name__ == '__main__':
    main()


##analysis from functional group paper -
#edge weight vs degree/prev figures
#graph alignment score
#


