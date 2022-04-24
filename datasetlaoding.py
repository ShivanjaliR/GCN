from itertools import combinations
from pathlib import Path

import nltk
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import numpy as np
import pandas as pd
import logging
from collections import OrderedDict
import re
import math
from nltk.corpus import stopwords
import os
import pickle
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('omw-1.4')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def nCr(n, r):
    f = math.factorial
    return int(f(n) / (f(r) * f(n - r)))


def word_word_edges(p_ij):
    word_word = []
    cols = list(p_ij.columns)
    cols = [str(w) for w in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1, w2] > 0):
            word_word.append((w1, w2, {"weight": p_ij.loc[w1, w2]}))
    return word_word


def save_as_pickle(filename, data):
    completeName = os.path.join("./data/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


class Dataset:

    class Document:
        def __init__(self):
            self.docName = ''
            self.tftdf = []
            self.UniqueWords = []
            self.noUniqueWords = 0
            self.wordFrequency = {}
            self.content = ""
            self.labels = []
            self.graph = nx.Graph()

        def setDocName(self, docName):
            self.docName = docName

        def getDocName(self):
            return self.docName

        def setTFIDF(self, tftdf):
            self.tftdf = tftdf

        def getTFIDF(self):
            return self.tftdf

        def setUniqueWords(self, uniqueWords):
            self.UniqueWords = uniqueWords

        def getUniqueWords(self):
            return self.UniqueWords

        def setNoUniqueWords(self, noUniqueWords):
            self.noUniqueWords = noUniqueWords

        def getNoUniqueWords(self):
            return self.noUniqueWords

        def setWordFrequency(self, wordFrequency):
            self.wordFrequency = wordFrequency

        def getWordFrequency(self):
            return self.wordFrequency

        def setContent(self, content):
            self.content = content

        def getContent(self):
            return self.content

        def setGraph(self, graph):
            self.graph = graph

        def getGraph(self):
            return self.graph

        def setLabels(self, labels):
            self.labels = labels

        def getLabels(self):
            return self.labels

    def __init__(self):
        self.document = Dataset.Document()

    def readFiles(self, features):
        dataset = Dataset()
        docs = []
        all_docs = {}
        # Reading Dataset files
        source_dir = Path('paper/')
        files = source_dir.iterdir()
        no_of_docs = 0
        en_stops = set(stopwords.words('english'))

        all_content = []
        all_content_line = ""
        index_doc = {}
        file_Id = 0
        doc_nodes = []
        lemmatizer = WordNetLemmatizer()
        for file in files:
            with open(file, encoding="utf-8") as fp:
                no_of_docs = no_of_docs + 1
                doc_nodes.append(str(file.name))
                res = [i for i in features if i in str(file.name).lower().replace('_', ' ')]
                index_doc[file_Id] = res[0]
                file_Id = file_Id + 1
                # Reading file content
                current_content = fp.read()
                content_no_spchar = re.sub(r"[^a-zA-Z]", " ", current_content).split()
                list_without_stop_word = ""
                for word in content_no_spchar:
                    if word.lower() not in en_stops and word.lower() != "" and word.lower() != " " and len(
                            word.lower()) > 3:
                        list_without_stop_word = list_without_stop_word + " " + lemmatizer.lemmatize(word.lower())
                all_content.append(list_without_stop_word)
                all_content_line = all_content_line + list_without_stop_word

        content_no_spchar = all_content
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(content_no_spchar)
        uniqueWords = vectorizer.get_feature_names_out()
        vocab = vectorizer.get_feature_names()
        tfidf = X.todense()
        df_tfidf = pd.DataFrame(X.toarray(), columns=np.array(vocab), index=np.array(doc_nodes))
        df = df_tfidf
        content = pd.DataFrame()
        # df[df.col != 0] = 1
        cnt = 0
        word_doc = {}
        for column in df_tfidf:
            column_values = df_tfidf[column].values
            non_zero_index = np.array((column_values).nonzero())
            if len(non_zero_index[0]) == 1:
                word_doc[column] = index_doc[non_zero_index[0][0]]
            else:
                non_zero_index_values = pd.Series(column_values).iloc[non_zero_index[0]]
                maxValue = max(non_zero_index_values)
                maxIndex = list(column_values).index(maxValue)
                word_doc[column] = index_doc[maxIndex]

        # Word Frequency and word occurance Frequency
        word2index = OrderedDict((word, index) for index, word in enumerate(vocab))
        occ_wordDict = np.zeros((len(vocab), len(vocab)), dtype=np.int32)
        wordDict = {}

        bigrams_word_dict = {}
        bigrams_wordDict = np.zeros((len(vocab), len(vocab)), dtype=np.int32)
        tokens = all_content_line.split()
        for i in range(len(tokens)):
            first_word = tokens[i].lower()
            if vocab.__contains__(first_word) == True:
                word_i = word2index[first_word]
                if first_word in wordDict.keys():
                    wordDict[first_word] = wordDict[first_word] + 1
                else:
                    wordDict[first_word] = 1

                if i + 1 > len(tokens) - 1:
                    break
                second_word = tokens[i + 1].lower()
                if vocab.__contains__(second_word) == True:
                    word_j = word2index[second_word]
                    bigrams = first_word + " " + second_word
                    if bigrams_wordDict[word_i][word_j] == 0:
                        bigrams_wordDict[word_i][word_j] = 1
                    else:
                        bigrams_wordDict[word_i][word_j] = bigrams_wordDict[word_i][word_j] + 1
                    if bigrams in bigrams_word_dict.keys():
                        bigrams_word_dict[bigrams] = bigrams_word_dict[bigrams] + 1
                    else:
                        bigrams_word_dict[bigrams] = 1

        # PMI Calculation
        p_ij = pd.DataFrame(bigrams_wordDict, index=vocab, columns=vocab)
        for col in p_ij.columns:
            p_ij[col] = p_ij[col] / wordDict[col]
        for row in p_ij.index:
            p_ij.loc[row, :] = p_ij.loc[row, :] / wordDict[row]
        p_ij = p_ij + 1E-9
        for col in p_ij.columns:
            p_ij[col] = p_ij[col].apply(lambda x: math.log(x))

        doc = Dataset.Document()
        doc.setDocName(file.name)
        doc.setWordFrequency(tfidf)
        doc.setUniqueWords(uniqueWords)
        doc.setWordFrequency(wordDict)
        doc.setNoUniqueWords(tfidf.shape[1])
        doc.setContent(content_no_spchar)

        words_edges = word_word_edges(p_ij)

        docs.append(doc)

        logger.info("Building graph (No. of document, word nodes: %d, %d)..." % (len(df_tfidf.index), len(vocab)))
        G = nx.Graph()
        logger.info("Adding document nodes to graph...")
        G.add_nodes_from(df_tfidf.index)  ## document nodes
        logger.info("Adding word nodes to graph...")
        G.add_nodes_from(vocab)  ## word nodes
        doc_word_edges = [(doc, word, {"weight": df_tfidf.loc[doc, word]}) for doc in df_tfidf.index
                          for word in df_tfidf.columns if df_tfidf.loc[doc, word] != 0]
        G.add_edges_from(doc_word_edges, color='black', weight=1)
        G.add_edges_from(words_edges, color='r', weight=2)
        if not os.listdir("./data/"):
            save_as_pickle("word_word_edges2.pkl", words_edges)
            save_as_pickle("text_graph2.pkl", G)
            logger.info('Created Graph Saved...')
        else:
            print('File is already saved...')
            logger.info('Graph is already Saved...')

        for n in G.nodes():
            G.nodes[n]['color'] = 'g' if n in df_tfidf.index else 'b'

        colors = nx.get_edge_attributes(G, 'color').values()
        pos = nx.spring_layout(G)
        node_colors = [node[1]['color'] for node in G.nodes(data=True)]
        plt.figure(figsize=(100, 100))
        nx.draw_networkx(G, pos,
                         edge_color=colors,
                         with_labels=True,
                         node_color=node_colors)
        plt.show()

        return word_doc, index_doc

    def generateLabels(self):
        source_dir = Path('classes/')
        files = source_dir.iterdir()
        features = []
        for file in files:
            with open(file, encoding="utf-8") as fp:
                classes = fp.read().split("\n")
                for feature in classes:
                    features.append(feature)
        return features

    def encodeLabeles(self, word_class, index_doc):
        classes_dict = {feature: np.identity(len(index_doc))[index, :] for index, feature in
                        enumerate(index_doc.values())}
        labels_onehot = np.array(list(map(classes_dict.get, word_class)),
                                 dtype=np.int32)
        return labels_onehot
