# This is a sample Python script.

import mat73
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import networkx as nx
import umap
import plotly.express as px
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import zscore
import cv2

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
        #print("on chunk number " +str(i) + " : " + str(round((i/num_chunks), 3)*100) + "%")
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
    graph = np.zeros((184, 184))

    for i in range(184):
        for j in range(184):
            graph[i][j] = compute_conmi(data[:, i], data[:, j])

    graph[np.isnan(graph)] = 0
    #graph[corr < 0] = 0
    np.fill_diagonal(graph, 0)


    return graph

def compute_graph_centrality(graph):
    deg_centrality = nx.degree_centrality(graph)

    plt.plot(*zip(*sorted(deg_centrality.items())))
    plt.title("Degree Centrality ")
    plt.show()

    close_centrality = nx.closeness_centrality(graph)


    plt.plot(*zip(*sorted(close_centrality.items())))
    plt.title("Close Centrality" )
    plt.show()

    bet_centrality = nx.betweenness_centrality(graph, normalized=False, endpoints=False)

    plt.plot(*zip(*sorted(bet_centrality.items())))
    plt.title("Between Centrality")
    plt.show()

    #pr = nx.pagerank(graph, alpha=0.8)

    #plt.plot(*zip(*sorted(pr.items())))
    #plt.title("Page Rank")
    #plt.show()

    return bet_centrality


def plot_edge_dist(G):
    edge_weights = []

    for u, v, w in G.edges(data=True):
        edge_weights.append(float(w['weight']))

    # plt.hist(degrees)
    # plt.show()
    n, bins, patches = plt.hist(edge_weights, bins=20, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.8, alpha=0.9)

    n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
    for i in range(len(patches)):
        patches[i].set_facecolor(cmap(n[i] / max(n)))  # Make one bin stand out

    print('mean: ')
    print(np.array(edge_weights).mean())

    plt.axvline(np.array(edge_weights).mean(), color='k', linestyle='dashed', linewidth=1)

    #patches[4].set_fc('red')  # Set color
    #patches[4].set_alpha(1)  # Set opacity# Add annotation
    #plt.annotate('Important Bar!', xy=(0.57, 175), xytext=(2, 130), fontsize=15,
    #             arrowprops={'width': 0.4, 'headwidth': 7,
    #                         'color': '#333333'})  # Add title and labels with custom font sizes

    plt.title('Non-reach Deep Neurons Edge Distribution', fontsize=12)
    plt.xlabel('Bins', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.show()


def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    #plt.hist(degrees)
    #plt.show()
    n, bins, patches = plt.hist(degrees, bins=40, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.8, alpha=0.9)


    n = n.astype('int')  # it MUST be integer# Good old loop. Choose colormap of your taste
    for i in range(len(patches)):
        patches[i].set_facecolor(cmap(n[i] / max(n)))  # Make one bin stand out

    #plt.axvline(degrees.mean(), color='k', linestyle='dashed', linewidth=1)

    patches[4].set_fc('red')  # Set color
    patches[4].set_alpha(1)  # Set opacity# Add annotation
    plt.annotate('Important Bar!', xy=(0.57, 175), xytext=(2, 130), fontsize=15,
                 arrowprops={'width': 0.4, 'headwidth': 7,
                             'color': '#333333'})  # Add title and labels with custom font sizes

    plt.title('S-D Full Session Degree Distribution', fontsize=12)
    plt.xlabel('Bins', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.show()

def plot_weight_dist(G):
    weights = [G.weight(n) for n in G.nodes()]
    plt.hist(weights)
    plt.show()

def show_graph_with_labels(adjacency_matrix, counter=0):

    upper_quartile = True

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
        plt.title("Non-reach Deep Neurons Graph (95th Percentile)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        plt.colorbar(scalarmappaple, cax=cax)
        nx.draw(G, layout, ax=ax, node_size=sizes, node_color=sizes, cmap=colormap)
        plt.show()


    return G

def compute_umap(data, counter=0):
    neural_data = np.array(data)[:, :]


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
        pass

def loop_through_reaches(reaches):
    dc_df = None
    init = True
    save_array=[]
    for idx, reach in enumerate(reaches):
        if(reach.shape[0] > 6):
            for i in range(2):
                mid = int(reach.shape[0]/2)

                if i == 0:
                    half_reach = reach[0:mid]
                elif i == 1:
                    half_reach = reach[mid:reach.shape[0]]

                #print(reach)
                df = pd.DataFrame(half_reach)
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

                graph = show_graph_with_labels(graph_adjacency_matrix, i)

                deg_centrality = nx.degree_centrality(graph)
                if not all(value == 0 for value in deg_centrality.values()):
                    save_array.append(idx)


                    dc = compute_graph_centrality(graph, i)
                    if init:
                        init = False
                        dc_df = pd.DataFrame.from_dict(dc, orient='index')
                        dc_df = dc_df.rename(columns={0: 'reach 0'})
                    else:
                        dc_df["reach" + str(idx)] = pd.Series(dc)

                    dc_df.to_csv('/home/macleanlab/Desktop/chris_data_out/centrality_csv/'+str(idx)+'.csv')
                    #compute_umap(np.array(df), idx)
                    pca = PCA(n_components=2)
                    plt.imshow(graph_adjacency_matrix)
                    plt.show()

                    principalComponents = pca.fit_transform(graph_adjacency_matrix)

                    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
                    plt.title("Stage: " + str(i) + " " + str(idx))
                    plt.show()
                    plot_degree_dist(graph)
                    np.save("/home/macleanlab/Downloads/idx_list.npy", np.array(save_array))

def loop_throgh_non_reaches():
    print()

def subset_reaches(data, masks):
    return data[np.where(masks != 0)[0], :] ##!= gets all reaches

def subset_non_reaches(data, masks):
    return data[np.where(masks == 0)[0], :] ##!= gets all reaches

'''
0.010869565217391353 - D-NR
0.06935255198487711 - D-R
0.010869565217391353 - D-FS
0.010869565217391353 - S-FS
0.07974952741020791 - S-R
0.010869565217391353 - S-NR
0.010869565217391353 - FS
0.010869565217391353 - NR
0.09085538752362954 - R

0.7977020321361059 - S-D Full
'''

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

    #data_dict = mat73.loadmat('/home/macleanlab/Downloads/evaluation_mouse98_0415/evaluated_mouse98_20220415_20230831-172942.mat')
    reach_masks = pd.read_csv('/home/macleanlab/Downloads/20220415_mouse98_allevents_cam1DLC_processed_reachlogical_30Hz.csv')
    reach_begin_end_indices = pd.read_csv('/home/macleanlab/Downloads/20220415_mouse98_allevents_cam1DLC_processed_reachindices_30Hz.csv')
    # print(data_dict)
    #neural_data = data_dict['neuron']['C']
    ##print(neural_data.shape)





    data_dict = mat73.loadmat('/home/macleanlab/Downloads/evaluation_mouse98_0415/evaluated_mouse98_20220415_20230831-172942.mat')
    spatial_data = mat73.loadmat('/home/macleanlab/Downloads/evaluation_mouse98_0415/spatialfootprintsforcellreg_mouse98_20220415.mat')

    data_dict_second_level = data_dict['neuron']

    print('here')

    spatial_data_second_level = spatial_data['A']
    print(spatial_data_second_level.shape)


    spikes = np.array(data_dict_second_level['S'])



    spatial_coordinates = dict()
    plt.imshow(spatial_data_second_level[100, :, :])
    plt.show()

    img = spatial_data_second_level[100, :, :].reshape((198, 318, 1))
    print(img.shape)

    for idx, im in enumerate(spatial_data_second_level):
        img = im.reshape((198, 318, 1))
        # convert the image to grayscale
        gray_image = np.uint8(img * 255)

        # convert the grayscale image to binary image
        ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

        # find contours in the binary image
        im2, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in im2:
            # calculate moments for each contour
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX)
            print(cY)
            spatial_coordinates[idx] = cY

    print(spatial_coordinates)

    sorted_spatial_coordinates = {k: v for k, v in sorted(spatial_coordinates.items(), key=lambda item: item[1])}
    print('sorted spatial coords')
    print(sorted_spatial_coordinates)
    print(np.array(list(sorted_spatial_coordinates.items())).shape)
    print(list(sorted_spatial_coordinates.items())[len(sorted_spatial_coordinates) // 2:])



    superficial_neurons = dict(list(sorted_spatial_coordinates.items())[len(sorted_spatial_coordinates) // 2:])
    deep_neurons = dict(list(sorted_spatial_coordinates.items())[:len(sorted_spatial_coordinates) // 2])

    print('------------------------------------------------------------------')
    print(superficial_neurons)
    print('deep')
    print(deep_neurons)
    print('------------------------------------------------------------------')

    deep_idx = list(deep_neurons.keys())
    superficial_idx = list(superficial_neurons.keys())



    df = create_pandas_df_transpose(spikes)

    df = pd.DataFrame(subset_reaches(df.to_numpy(), reach_masks))

    print('spikes: ')
    print(spikes.shape)


    print(df)

    idx = np.random.randint(low=0, high=60000, size=10000)
    print(np.sort(idx))

    spikes = df.sample(10000, random_state=69).to_numpy()
    print(spikes)



    mu = np.mean(df.to_numpy())
    sigma = np.std(df.to_numpy())
    threshold = mu+sigma
    print(threshold)
    spikes[spikes >= threshold] = 1
    spikes[spikes < threshold] = 0

    print(spikes[spikes==1])


    graph_adjacency_matrix = create_graph(spikes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    axp = ax.imshow(graph_adjacency_matrix, cmap='inferno')
    cb = plt.colorbar(axp, ax=[ax], location='right')
    plt.title("Full Session Weight Distribution")
    plt.show()


    background_graph = background(graph_adjacency_matrix)
    residual_graph = residual(background_graph, graph_adjacency_matrix)
    normed_graph = normed_residual(residual_graph)


    #normed_graph = np.load("/home/macleanlab/Desktop/chris_data_out/numpy/normed_graph.npy")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    axp = ax.imshow(normed_graph, cmap='inferno')
    cb = plt.colorbar(axp, ax=[ax], location='right')
    plt.title("Reach Weight Distribution")
    plt.show()


    graph = show_graph_with_labels(normed_graph)

    edgelist = list(graph.edges(data=True))

    for u,v,a in edgelist:
        ##if connect between super/deep keep otherwise remove edge between u and v
        if (u in deep_idx and v in superficial_idx) or (u in deep_idx and v in superficial_idx):
            print("Found deep-superficial edge")
        else:
            graph.remove_edge(u, v)

    graph.remove_nodes_from(list(nx.isolates(graph)))


    compute_umap(normed_graph)

    pca = PCA(n_components=2)




    deg = compute_graph_centrality(graph)

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

    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(7)

    plt.plot(deg_z_scores_to_plot[153:183], new[153:183], 'g-')
    plt.xlabel("Standardized Z-Score")
    plt.ylabel("Neuron Index")
    plt.title("S-D Full Session Betweenness Centrality Top N")
    plt.yticks(fontsize=10)
    plt.show()




    adj_matrix = nx.adjacency_matrix(graph).toarray()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    axp = ax.imshow(adj_matrix, cmap='inferno')
    cb = plt.colorbar(axp, ax=[ax], location='right')
    plt.title("S-D Full Session Weight Distribution")
    plt.show()


    principalComponents = pca.fit_transform(adj_matrix)
    print(principalComponents)
    print(principalComponents.shape)
    print(pca.explained_variance_ratio_)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.show()
    plot_degree_dist(graph)

    sparsity = 1.0 - (np.count_nonzero(adj_matrix) / float(adj_matrix.size))
    print('sparsity: ')
    print(sparsity)
    print('*************************************************************************')

    '''


    #np.save("/home/macleanlab/Desktop/chris_data_out/numpy/normed_graph.npy", normed_graph)


    deg = compute_graph_centrality(graph)
    compute_umap(normed_graph)

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(graph_adjacency_matrix)
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
    plt.title("Reach Degree Centrality")
    plt.show()


    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(7)


    plt.plot(deg_z_scores_to_plot[153:183], new[153:183], 'g-')
    plt.xlabel("Standardized Z-Score")
    plt.ylabel("Neuron Index")
    plt.title("Reach Degree Centrality Top N")
    plt.yticks(fontsize=10)
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








    ##look at autocorrelation between all reaches - csv file of centrality or maybe between adj matrices of graphs
    ##analysis of first and second half of reaches - centrality and autocorrelation

    ##arr = np.load("/home/macleanlab/Downloads/idx_list.npy")

    spatial_coordinates = dict()
    plt.imshow(spatial_data_second_level[100, :, :])
    plt.show()

    img = spatial_data_second_level[100, :, :].reshape((198, 318, 1))
    print(img.shape)

    for idx, im in enumerate(spatial_data_second_level):
        img = im.reshape((198, 318, 1))
        # convert the image to grayscale
        gray_image = np.uint8(img * 255)

        # convert the grayscale image to binary image
        ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

        # find contours in the binary image
        im2, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in im2:
            # calculate moments for each contour
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX)
            print(cY)
            spatial_coordinates[idx] = cY

    print(spatial_coordinates)

    sorted_spatial_coordinates = {k: v for k, v in sorted(spatial_coordinates.items(), key=lambda item: item[1])}
    print('sorted spatial coords')
    print(sorted_spatial_coordinates)
    print(np.array(list(sorted_spatial_coordinates.items())).shape)
    print(list(sorted_spatial_coordinates.items())[len(sorted_spatial_coordinates) // 2:])



    superficial_neurons = dict(list(sorted_spatial_coordinates.items())[len(sorted_spatial_coordinates) // 2:])
    deep_neurons = dict(list(sorted_spatial_coordinates.items())[:len(sorted_spatial_coordinates) // 2])

    print('------------------------------------------------------------------')
    print(superficial_neurons)
    print('deep')
    print(deep_neurons)
    print('------------------------------------------------------------------')

    deep_idx = list(deep_neurons.keys())
    superficial_idx = list(superficial_neurons.keys())

    spikes_deep = np.take(spikes, deep_idx, 0)
    spikes_superficial = np.take(spikes, superficial_idx, 0)

    print(deep_idx)


    print('deep shape')
    print(spikes_deep.shape)
    print('superficial shape')
    print(spikes_superficial.shape)

    df = create_pandas_df_transpose(spikes)
    df = pd.DataFrame(subset_reaches(df.to_numpy(), reach_masks))

    spikes_superficial = df.to_numpy()

    mu = np.mean(df.to_numpy())
    sigma = np.std(df.to_numpy())
    print(mu)
    print(sigma)
    threshold = .99
    print(threshold)
    spikes_superficial[spikes_superficial >= threshold] = 1
    spikes_superficial[spikes_superficial < threshold] = 0

    print('how many zeros')
    print(spikes_superficial[spikes_superficial==0].shape)
    print('how many ones')
    print(spikes_superficial[spikes_superficial==1].shape)

    
    #deep -
    
    #how many zeros
    #(5360525,)
    #how many ones
    #(298395,)
    
    #superficial -
    #how many zeros
    #(5387495,)
    #how many ones
    #(271425,)
    

    graph_adjacency_matrix = create_graph(spikes_superficial)

    sparsity = 1.0 - (np.count_nonzero(graph_adjacency_matrix) / float(graph_adjacency_matrix.size))
    print('sparsity: ')
    print(sparsity)
    print('*************************************************************************')

    background_graph = background(graph_adjacency_matrix)
    residual_graph = residual(background_graph, graph_adjacency_matrix)
    normed_graph = normed_residual(residual_graph)
    print('here')
    graph = show_graph_with_labels(normed_graph)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    axp = ax.imshow(normed_graph, cmap='inferno')
    cb = plt.colorbar(axp, ax=[ax], location='right')
    plt.title("Non-reach Deep Neurons Weight Distribution")
    plt.show()

    deg = compute_graph_centrality(graph)

    print('deg')
    print(deg)

    deg = dict(zip(superficial_idx, list(deg.values())))

    print('deg')
    print(deg)


    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(spikes_superficial) #spikes_superficial
    print(principalComponents)
    print(principalComponents.shape)
    print(pca.explained_variance_ratio_)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.show()
    plot_degree_dist(graph)
    plot_edge_dist(graph)

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
    plt.title("Reach Deep Neurons Degree Centrality")
    plt.show()

    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(7)

    plt.plot(deg_z_scores_to_plot[72:92], new[72:92], 'g-')
    plt.xlabel("Standardized Z-Score")
    plt.ylabel("Neuron Index")
    plt.title("Reach Superficial Neurons Degree Centrality Top N")
    plt.yticks(fontsize=10)
    plt.show()





##finished deep and super for whole session - now look at super and deep for reach non/reach
##then quantify overlap of central nodes
##then look into pca and start practicing presentation





    #reach_list = subset_reaches_by_frame_start_and_end(df.to_numpy(), reach_begin_end_indices.to_numpy())
    #print(reach_list)
    #loop_through_reaches(reach_list)

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


