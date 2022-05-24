'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''

from pathlib import Path
from resources.constants import output_folder, text_graph_name, input_folder, \
    dataset_details, output_column_filename, output_column_noOfWords, output_column_content, summary_column_noOfFiles, \
    summary_column_noOfUniqueWords, summary_column_uniqueWords, log_save_graph, log_pkl_saved, log_add_doc_node, \
    log_building_graph, text_graph_pkl_file_name, word_edge_graph_pkl_file_name, graph_document_edges, graph_no_nodes, \
    graph_word_edges, graph_no_edges, graph_document_nodes, graph_word_nodes, graph_no_document_nodes, \
    graph_no_word_nodes, graph_no_document_edges, graph_no_word_edges, graph_details
from utils import save_as_pickle, word_word_edges
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
import re
import math
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from constants import *
import csv

nltk.download('wordnet')
nltk.download('omw-1.4')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Dataset:

    class Graph:
        """
        Graph Information and its respective attributes
        """

        def __init__(self):
            self.doc_nodes = []
            self.no_doc_nodes = 0
            self.word_nodes = []
            self.no_word_nodes = 0
            self.doc_to_word_edges = []
            self.no_doc_to_word_edges = 0
            self.word_to_word_edges = []
            self.no_word_to_word_edges = 0
            self.total_nodes = 0
            self.total_edges = 0

        def setDocNodes(self, doc_nodes):
            """
            Set Document nodes
            :param doc_nodes:
            :return: None
            """
            self.doc_nodes = doc_nodes

        def getDocNodes(self):
            """
            Return document nodes
            :return: doc_nodes
            """
            return self.doc_nodes

        def setNoDocNodes(self, noOfDocNodes):
            """
            Set No of document nodes
            :param noOfDocNodes:
            :return: None
            """
            self.no_doc_nodes = noOfDocNodes

        def getNoDocNodes(self):
            """
            Get No of document nodes
            :return: no_doc_nodes
            """
            return self.no_doc_nodes

        def setWordNodes(self, word_nodes):
            """
            Set Word Nodes
            :param word_nodes:
            :return: None
            """
            self.word_nodes = word_nodes

        def getWordNodes(self):
            """
            Get Word Nodes
            :return: word_nodes
            """
            return self.word_nodes

        def setNoWordNodes(self, no_word_nodes):
            """
            Set No of word nodes
            :param no_word_nodes:
            :return: None
            """
            self.no_word_nodes = no_word_nodes

        def getNoWordNodes(self):
            """
            Get No of word Nodes
            :return: no_word_nodes
            """
            return self.no_word_nodes

        def setTotalNodes(self, totalNodes):
            """
            Set Total Number of Nodes
            :param totalNodes:
            :return: None
            """
            self.total_nodes = totalNodes

        def getTotalNodes(self):
            """
            Get Total number of nodes
            :return: total_nodes
            """
            return self.total_nodes

        def setDocWordEdges(self, doc_to_word_edges):
            """
            Set Document to word edges
            :param doc_to_word_edges:
            :return: None
            """
            self.doc_to_word_edges = doc_to_word_edges

        def getDocWordEdges(self):
            """
            Get Dcument to Word edges
            :return: doc_to_word_edges
            """
            return self.doc_to_word_edges

        def setNoDocWordEdges(self, no_doc_to_word_edges):
            """
            Set Number of Document to word edges
            :param no_doc_to_word_edges:
            :return: None
            """
            self.no_doc_to_word_edges = no_doc_to_word_edges

        def getNoDocWordEdges(self):
            """
            Get Number of Document to Word Edges
            :return: no_doc_to_word_edges
            """
            return self.no_doc_to_word_edges

        def setWordWordEdges(self, word_to_word_edges):
            """
            Set Word to Word Edges
            :param word_to_word_edges:
            :return: None
            """
            self.word_to_word_edges = word_to_word_edges

        def getWordWordEdges(self):
            """
            Get Word to word edges
            :return: word_to_word_edges
            """
            return self.word_to_word_edges

        def setNoWordWordEdges(self, no_word_to_word_edges):
            """
            Set Number of Word to word edges
            :param no_word_to_word_edges:
            :return: None
            """
            self.no_word_to_word_edges = no_word_to_word_edges

        def getNoWordWordEdges(self):
            """
            Get Number of Word to word edges
            :return: no_word_to_word_edges
            """
            return self.no_word_to_word_edges

        def setTotalEdges(self, total_edges):
            """
            Set Total number of edges
            :param total_edges:
            :return: None
            """
            self.total_edges = total_edges

        def getTotalEdges(self):
            """
            Get total number of edges
            :return: total_edges
            """
            return self.total_edges

    class Document:
        """
         Class represents Document.

         Attributes:
         -----------
         docName: String
                Document Name

         Methods:
         ---------
         setDocName(self, docName)
            Set document name

         getDocName(self)
            Get document name

        """

        def __init__(self):
            self.docName = ''
            self.words = []
            self.noOfWords = 0

        def setDocName(self, docName):
            """
            Set Document Name
            :param docName:
            :return: None
            """
            self.docName = docName

        def getDocName(self):
            """
            Get Document Name
            :return: docName
            """
            return self.docName

        def setWords(self, words):
            """
            Set all words present in given document
            :param words:
            :return: None
            """
            self.words = words

        def getWords(self):
            """
            Get all words present in the given document
            :return: words
            """
            return self.words

        def setNoOfWords(self, noOfWords):
            """
            Set number of words in given document
            :param noOfWords:
            :return: noOfWords
            """
            self.noOfWords = noOfWords

        def getNoOfWords(self):
            """
            Get number of words present in given document
            :return:
            """
            return self.noOfWords

    def __init__(self):
        """
         Class represents Dataset and its attributes

         Attributes:
         ------------
         documents: Array
                List of Document class i.e list of documents.
         cleanContent: String
                File content without special characters, numbers, stop words, removing strings less that three characters
                with lemmatization.
         all_content_array: Array
                String of array with all file content.
         noOfDocs: Number
                Number of documents.
         index_doc: List
                Document with respective its index.
         fileName: Array
                Array of file names
         tfidf: DataFrame
                Matrix of word and its respective document with its TF-IDF(Term Frequency–Inverse Document Frequency)
                calculated value.
         pmiCnt: DataFrame
                Matrix of co-occurred words with its respective calculated PMI(Point-wise Mutual Information) value.
         featureName: Array
                Array of features/classes names
        """
        self.documents = []
        self.cleanContent = ''
        self.totalUniqueWords = 0
        self.uniqueWords = []
        self.all_content_line = ''
        self.all_content_array = []
        self.noOfDocs = 0
        self.index_doc = {}
        self.fileNames = []
        self.tfidf = pd.DataFrame()
        self.pmiCnt = pd.DataFrame()
        self.featureNames = []
        self.graph = Dataset.Graph()

    def setnoOfDocs(self, noOfDocs):
        """
        Set Number of Documents in the Dataset
        :param noOfDocs: Number of Documents
        :return: None
        """
        self.noOfDocs = noOfDocs

    def getnoOfDocs(self):
        """
        Get Number of documents in the Dataset
        :return: noOfDocs
        """
        return self.noOfDocs

    def setCleanContent(self, cleanContent):
        """
        Set file content without special characaters, numbers, stop words, removing strings less than three characters
        with lemmatization
        :param cleanContent:
        :return: None
        """
        self.cleanContent = cleanContent

    def getCleanContent(self):
        """
        Get clean file content
        :return: cleanContent
        """
        return self.cleanContent

    def setUniqueWords(self, uniqueWords):
        """
        Set unique words from dataset
        :param uniqueWords:
        :return: None
        """
        self.uniqueWords = uniqueWords

    def getUniqueWords(self):
        """
        Get unique words from dataset
        :return: uniqueWords
        """
        return self.uniqueWords

    def setNoUniqueWords(self, uniqueWords):
        """
        Set Unique words
        :param uniqueWords:
        :return: None
        """
        self.totalUniqueWords = uniqueWords

    def getNoUniqueWords(self):
        """
        Get total number of unique words from dataset
        :return: totalWords
        """
        return self.totalUniqueWords

    def setAllContentLine(self, all_content_line):
        """
        Set all files content in one string form
        :param all_content_line:
        :return: None
        """
        self.all_content_line = all_content_line

    def getAllContentLine(self):
        """
        Get all files content in one string form
        :return: all_content_line
        """
        return self.all_content_line

    def setAllContentArray(self, all_content_array):
        """
        Set all files content in array form
        :param all_content_array:
        :return: None
        """
        self.all_content_array = all_content_array

    def getAllContentArray(self):
        """
        Get all files content in array form
        :return: all_content_array
        """
        return self.all_content_array

    def setIndexDoc(self, index_doc):
        """
        Set document to index list
        :param index_doc:
        :return: None
        """
        self.index_doc = index_doc

    def getIndexDoc(self):
        """
        Get document to index list
        :return: index_doc
        """
        return self.index_doc

    def setDocuments(self, documents):
        """
        Set array of Documents
        :param documents:
        :return: None
        """
        self.documents = documents

    def getDocuments(self):
        """
        Get array of Documents
        :return: documents
        """
        return self.documents

    def setFileNames(self, fileNames):
        """
        Set array of file names
        :param fileNames:
        :return: None
        """
        self.fileNames = fileNames

    def getFileNames(self):
        """
        Get array of file names
        :return: fileNames
        """
        return self.fileNames

    def setTfidf(self, tfidf):
        """
        Set TF-IDF of all words in all files
        :param tfidf:
        :return: None
        """
        self.tfidf = tfidf

    def getTfidf(self):
        """
        Get TF-IDF of all words in all files
        :return:
        """
        return self.tfidf

    def setPmiCnt(self, pmiCnt):
        """
        Set PMI values of all co-occurred words
        :param pmiCnt:
        :return: None
        """
        self.pmiCnt = pmiCnt

    def getPmiCnt(self):
        """
        Get PMI values of all co-occurred words
        :return:
        """
        return self.pmiCnt

    def setfeatureNames(self, featureNames):
        """
        Set feature/class names
        :param featureNames:
        :return: None
        """
        self.featureNames = featureNames

    def getFeatureNames(self):
        """
        Get feature/class names
        :return: featureNames
        """
        return self.featureNames

    def setGraph(self, graph):
        """
        Set graph belongs to respective dataset
        :param graph:
        :return: None
        """
        self.graph = graph

    def getGraph(self):
        """
        Get Graph object of respective dataset
        :return: graph
        """
        return self.graph

    def readFilesDocCleaning(self, features):
        """
        Read input files and clean its content and preserved file content as per requirement.
        :param features: List of feature/class names
        :return: None
        """
        docs = []
        # Reading Dataset files
        source_dir = Path(input_folder)
        files = source_dir.iterdir()
        no_of_docs = 0

        # Set of stop words
        en_stops = set(stopwords.words('english'))

        all_content_array = []
        all_content_line = ""
        index_doc = {}
        file_Id = 0
        doc_nodes = []
        lemmatizer = WordNetLemmatizer()
        for file in files:
            with open(file, encoding="utf-8") as fp:
                no_of_docs = no_of_docs + 1
                # List file names as document nodes
                doc_nodes.append(str(file.name))

                # Cleaned file name
                cleanedFileName = [i for i in features if i in str(file.name).lower().replace('_', ' ')]
                index_doc[file_Id] = cleanedFileName[0]
                file_Id = file_Id + 1
                # Reading file content
                current_content = fp.read()

                # Removing spaces, special characters from tokens
                content_no_spchar = re.sub(r"[^a-zA-Z]", " ", current_content).split()

                # Set Document class object
                doc = Dataset.Document()
                doc.setDocName(file.name)
                doc.setWords(content_no_spchar)
                doc.setNoOfWords(len(content_no_spchar))
                docs.append(doc)  # Keep track of document list

                list_without_stop_word = ""
                for word in content_no_spchar:
                    # Removing stop words, space and string less than three characters
                    if word.lower() not in en_stops and word.lower() != "" and word.lower() != " " and len(
                            word.lower()) > 3:
                        list_without_stop_word = list_without_stop_word + " " + lemmatizer.lemmatize(word.lower())
                all_content_array.append(list_without_stop_word)
                all_content_line = all_content_line + list_without_stop_word

                # Set all resultant values to dataset class object
                self.setCleanContent(content_no_spchar)
                self.setAllContentLine(all_content_line)
                self.setAllContentArray(all_content_array)
                self.setIndexDoc(index_doc)
                self.setDocuments(docs)
                self.setFileNames(doc_nodes)
                self.setnoOfDocs(len(docs))

    def getDatasetDetails(self):
        """
        Get Dataset Details.
        :return: None
        """
        documents = self.getDocuments()
        table = []
        row = [output_column_filename, output_column_noOfWords, output_column_content]
        table.append(row)
        for document in documents:
            row = []
            row.append(document.getDocName())
            row.append(document.getNoOfWords())
            row.append(document.getWords())
            table.append(row)

        summary = []
        row = []
        row.append(summary_column_noOfFiles)
        row.append(self.getnoOfDocs())
        summary.append(row)
        row = []
        row.append(summary_column_noOfUniqueWords)
        row.append(self.getNoUniqueWords())
        summary.append(row)
        row = []
        row.append(summary_column_uniqueWords)
        row.append(self.getUniqueWords())
        summary.append(row)

        with open(dataset_details, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in table]
            [writer.writerow(r) for r in summary]

    def getGraphDetails(self):
        """
        Write Graph details (No of nodes, No of edges etc.) in csv file
        :return: None
        """
        table = []
        row = [graph_no_document_nodes, graph_no_word_nodes, graph_no_nodes,
               graph_no_document_edges, graph_no_word_edges, graph_no_edges]
        table.append(row)
        graph = self.getGraph()
        row = [graph.getNoDocNodes(), graph.getNoWordNodes(), graph.getTotalNodes(),
               graph.getNoDocWordEdges(), graph.getNoWordWordEdges(), graph.getTotalEdges()]
        table.append(row)
        row = [graph_document_nodes]
        table.append(row)
        row = []
        for docNode in graph.getDocNodes():
            row.append(docNode)
        table.append(row)
        row = [graph_word_nodes]
        table.append(row)
        row = []
        for wordNode in graph.getWordNodes():
            row.append(wordNode)
        table.append(row)
        row = [graph_document_edges]
        table.append(row)
        row = []
        for docEdge in graph.getDocWordEdges():
            row.append(docEdge)
        table.append(row)
        row = [graph_word_edges]
        table.append(row)
        row = []
        for wordEdge in graph.getWordWordEdges():
            row.append(wordEdge)
        table.append(row)

        with open(graph_details, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in table]


    def FrequencyCalculation(self):
        """
        Calculate TF-IDF of all file content, Word frequency in all files, PMI of co-occurred words
        :return: None
        """
        # Dataset Setting Values
        content_no_spchar = self.getAllContentArray()
        all_content_line = self.getAllContentLine()

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(content_no_spchar)
        vocab = vectorizer.get_feature_names()
        self.setfeatureNames(vocab)
        tfidf = X.todense()
        df_tfidf = pd.DataFrame(X.toarray(), columns=np.array(vocab), index=np.array(self.getFileNames()))
        self.setTfidf(df_tfidf)

        self.setUniqueWords(vocab)
        self.setNoUniqueWords(vocab.__len__())

        # Word and its respective index
        word2index = OrderedDict((word, index) for index, word in enumerate(vocab))
        wordDict = {}

        # Bigram frequency
        bigrams_word_dict = {}
        bigrams_wordDict = np.zeros((len(vocab), len(vocab)), dtype=np.int32)
        tokens = all_content_line.split()
        for i in range(len(tokens)):
            first_word = tokens[i].lower()
            if vocab.__contains__(first_word) == True:
                word_i = word2index[first_word]
                # Word frequency
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

        self.setPmiCnt(p_ij)

    def createGraph(self):
        """
        Create graph from saved values
        Document are used as Document Nodes
        Words are used as Word Nodes
        TF-IDF of all unique words used for Word to Document edge
        PMI of co-occurred words used for Word to Word edge
        Save Graph in pickle file
        :return: None
        """
        graph = self.getGraph()
        logger.info(log_building_graph % (
            len(self.getTfidf().index), len(self.getFeatureNames())))
        G = nx.Graph()
        logger.info(log_add_doc_node)
        G.add_nodes_from(self.getTfidf().index)  # Document Nodes
        graph.setDocNodes(self.getTfidf().index)
        graph.setNoDocNodes(len(self.getTfidf().index))
        logger.info(log_add_doc_node)
        G.add_nodes_from(self.getFeatureNames())  # Word Nodes
        graph.setWordNodes(self.getFeatureNames())
        graph.setNoWordNodes(len(self.getFeatureNames()))
        graph.setTotalNodes(graph.getNoDocNodes() + graph.getNoWordNodes())

        # Document-to-Word edges
        doc_word_edges = [(doc, word, {"weight": self.getTfidf().loc[doc, word]}) for doc in self.getTfidf().index
                          for word in self.getTfidf().columns if self.getTfidf().loc[doc, word] != 0]
        G.add_edges_from(doc_word_edges, color='black', weight=1)
        graph.setDocWordEdges(doc_word_edges)
        graph.setNoDocWordEdges(len(doc_word_edges))

        # Word-to-Word Edges
        words_edges = word_word_edges(self.getPmiCnt())
        G.add_edges_from(words_edges, color='r', weight=2)
        graph.setWordWordEdges(words_edges)
        graph.setNoWordWordEdges(len(words_edges))

        graph.setTotalEdges(len(graph.getDocWordEdges()) + len(graph.getWordWordEdges()))
        if not os.listdir(output_folder):
            print(log_save_graph)
            save_as_pickle(word_edge_graph_pkl_file_name, words_edges)
            save_as_pickle(text_graph_pkl_file_name, G)
            logger.info(log_save_graph)
        else:
            print(log_pkl_saved)
            logger.info(log_pkl_saved)

        '''
           Coloring nodes.
           Document Nodes: Green Color
           Word Nodes: Blue Color
        '''
        for n in G.nodes():
            G.nodes[n]['color'] = 'g' if n in self.getTfidf().index else 'b'

        '''
           Fetching edge color attribute.
           Document-to-Word: Black colored edge 
           Word-to-Word: Red colored edge            
        '''
        colors = nx.get_edge_attributes(G, 'color').values()
        pos = nx.spring_layout(G)
        node_colors = [node[1]['color'] for node in G.nodes(data=True)]
        plt.figure(figsize=(100, 100))
        plt.title(text_graph_name)
        nx.draw_networkx(G, pos,
                         edge_color=colors,
                         with_labels=True,
                         node_color=node_colors)
        plt.savefig(text_graph_name, dpi=100)
        plt.show()

    def labelSetting(self):
        """
        Label words with its respective class (i.e in which document that word is occurred)
        :return: None

        """
        word_doc = {}
        for column in self.getTfidf():
            column_values = self.getTfidf()[column].values
            non_zero_index = np.array((column_values).nonzero())
            if len(non_zero_index[0]) == 1:
                word_doc[column] = self.getIndexDoc()[non_zero_index[0][0]]
            else:
                '''
                If same unique word occurred in more than one document then 
                consider highest TF-IDF value and label that word with that respective class.                
                '''
                non_zero_index_values = pd.Series(column_values).iloc[non_zero_index[0]]
                maxValue = max(non_zero_index_values)
                maxIndex = list(column_values).index(maxValue)
                word_doc[column] = self.getIndexDoc()[maxIndex]
        return word_doc
