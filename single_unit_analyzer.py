import numpy as np
import pandas as pd
import networkx as nx

class single_unit_analyzer:
    def __init__(self):
        print('alllll the single units, alllll the single units')
        self.df = pd.DataFrame()
        self.graph = None

    def compute_top_central_units(self):
        deg_centrality = nx.degree_centrality(self.graph)
        self.df = pd.DataFrame.from_dict(deg_centrality, orient='index', columns=["Degree"])

        new_col_array = []
        for i in range(60):
            new_col_array.append("Neuron idx " + str(i))

        self.df.insert(loc=0, column='Neuron IDX', value=new_col_array)

        close_centrality = nx.closeness_centrality(self.graph)
        self.df['Closeness'] = self. df.index.to_series().map(close_centrality)

        bet_centrality = nx.betweenness_centrality(self.graph, normalized=True, endpoints=False)

        self.df['Betweenness'] = self. df.index.to_series().map(bet_centrality)

        pr = nx.pagerank(self.graph, alpha=0.8)

        self.df['Page Rank'] = self. df.index.to_series().map(pr)


    def set_graph(self, graph):
        self.graph = graph

    def get_graph(self):
        return self.graph

    def set_df(self, df):
        self.df = df

    def get_df(self):
        return self.df