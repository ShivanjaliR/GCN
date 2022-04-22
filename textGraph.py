import os
import pickle
import numpy as np
import networkx as nx
import pandas as pd


class TextGraph:
    def __init__(self):
        self.num = 100

    def loadGraph(self):
        completeName = os.path.join("./data/", 'text_graph3.pkl')
        with open(completeName, 'rb') as pkl_file:
            G = pickle.load(pkl_file)

        # Building adjacency and degree matrices...
        A = nx.to_numpy_matrix(G, weight="weight")
        A = A + np.eye(G.number_of_nodes())

        dictionary = pd.DataFrame(A, columns=np.array(G.nodes))
        print(dictionary)

        degrees = []
        for d in G.degree():
            if d == 0:
                degrees.append(0)
            else:
                degrees.append(d[1] ** (-0.5))
        degrees = np.diag(degrees)  # Degree Matrix ^ (-1/2)
        X = np.eye(G.number_of_nodes())  # identity matrix
        A_hat = degrees @ A @ degrees  # A_hat = D^-1/2 A D^-1/2
        f = X
        return f, X, A_hat
