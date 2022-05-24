'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''

import os
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from resources.constants import output_folder,  text_graph_pkl_file_name

class TextGraph:
    """
     Class represents Text Graph and its related attributes
    """
    def __init__(self):
        self.num = 100
        self.no_of_nodes = 0
        self.nodes = []

    def loadGraph(self):
        """
        Load Graph from pickle file and fetch its respective attributes
        :return: f, X, A_hat
                 Identity Matrix, Normalized symmetric adjacency matrix
        """
        completeName = os.path.join(output_folder, text_graph_pkl_file_name)
        with open(completeName, 'rb') as pkl_file:
            G = pickle.load(pkl_file)

        # Building adjacency Matrix...
        A = nx.to_numpy_matrix(G, weight="weight")
        # Add Adjacency Matrix and Identity Matrix
        A = A + np.eye(G.number_of_nodes())

        self.setNoOfNodes(G.number_of_nodes())
        self.setNodes(G.nodes)


        dictionary = pd.DataFrame(A, columns=np.array(G.nodes))

        # Building Degree Matrix
        degrees = []
        for d in G.degree():
            if d == 0:
                degrees.append(0)
            else:
                degrees.append(d[1] ** (-0.5))
        degrees = np.diag(degrees)  # Degree Matrix ^ (-1/2)
        X = np.eye(G.number_of_nodes())  # identity matrix as Input
        A_hat = degrees @ A @ degrees  # A_hat = D^-1/2 A D^-1/2
        f = X
        return f, X, A_hat, G
