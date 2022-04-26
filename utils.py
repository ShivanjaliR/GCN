'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''

import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm
from itertools import combinations

from resources.constants import output_folder, resource_path


def save_as_pickle(filename, data):
    """
    Save Graph, Dataset, edges in pickle file
    :param filename: File name where you want to save graph/dataset/edges
    :param data: Data of graph/dataset/edges
    :return: None
    """
    completeName = os.path.join(output_folder, filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def plotGraph(x, y, x_label, y_label, graph_title):
    """
    Plot graph for respective data
    :param x: X-axis values
    :param y: Y-axis values
    :param x_label: X-axis labels
    :param y_label: Y-axis labels
    :param graph_title: Name of graph
    :return: None
    """
    plt.figure(figsize=(15, 4))
    plt.title(graph_title, fontdict={'fontweight': 'bold', 'fontsize': 18})
    plt.plot(x, y, label=graph_title)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(graph_title + 'png', dpi=100)
    plt.show()


def generateLabels():
    """
    Read unique classes/labels from the file
    :return: None
    """
    features = []
    '''source_dir = Path(resource_path)
    files = source_dir.iterdir()
    features = []
    for file in files:
        with open(file, encoding="utf-8") as fp:
            classes = fp.read().split("\n")
            for feature in classes:
                features.append(feature)'''
    f = open(resource_path, "r")
    classes = f.read().split("\n")
    for feature in classes:
        features.append(feature)
    return features


def nCr(n, r):
    """
    Calculate combinations
    :param n: Number of items in list
    :param r: Number of items suppose to select from n
    :return: Calculated result
    """
    f = math.factorial
    return int(f(n) / (f(r) * f(n - r)))


def word_word_edges(p_ij):
    """
    Create word-to-word edges and assign weights to respective edge
    :param p_ij: Words Co-occurrenced (Bigram) frequency
    :return: None
    """
    word_word = []
    cols = list(p_ij.columns)
    cols = [str(w) for w in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1, w2] > 0):
            word_word.append((w1, w2, {"weight": p_ij.loc[w1, w2]}))
    return word_word


def accuracy(output, labels):
    """
    Calculate accuracy between predicted result and actual result
    :param output: Predicted result from GCN classifier model
    :param labels: Actual labels
    :return: Number of correct labels/classes
    """
    _, prediction = output.max(1)
    prediction = prediction.numpy()
    actual_labels = [(label) for label in labels]
    correct = sum(actual_labels == prediction)
    return correct / len(prediction)


def encodeLabeles(word_class, index_doc):
    """
    Encode class labels in binary result
    :param word_class: word
    :param index_doc: Index of documents
    :return: One-hot encoding of classes
    """
    classes_dict = {feature: np.identity(len(index_doc))[index, :] for index, feature in
                    enumerate(index_doc.values())}
    labels_onehot = np.array(list(map(classes_dict.get, word_class)),
                             dtype=np.int32)
    return labels_onehot
