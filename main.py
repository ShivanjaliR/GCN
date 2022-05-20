'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''

from __future__ import division
from __future__ import print_function

import sys

from sklearn.model_selection import train_test_split
import numpy as np
from datasetlaoding import Dataset
import torch
from resources.constants import training_accuracy_plot_name, training_loss_plot_name, output_file, \
    training_loss_plot_file_name, \
    training_accuracy_plot_file_name, plot_x_axis, plot_y_axis_loss, plot_y_axis_accuracy, learning_rate, \
    num_of_epochs, model_filename, testing_accuracy_plot_file_name, testing_accuracy_plot_name, \
    testing_loss_plot_file_name, testing_loss_plot_name
from textGraph import TextGraph
from gcnmodel import gcn
import torch.optim as optim
from utils import plotGraph, accuracy, generateLabels
import pickle

if __name__ == '__main__':

    sys.stdout = open(output_file, "w")

    # Step 1: Dataset Generation and Cleaning
    dataset = Dataset()
    features = generateLabels()
    dataset.readFilesDocCleaning(features)
    index_doc = dataset.getIndexDoc()

    # Step 2: Frequency Calculation
    dataset.FrequencyCalculation()

    # Step 1.2: Dataset Details
    dataset.getDatasetDetails()

    # Step 3: Creating Graph
    dataset.createGraph()

    # Step 4: Labeling words
    wordClasses = dataset.labelSetting()

    node_labels_values = list(index_doc.values())

    node_labels = []
    for node_label in node_labels_values:
        if node_label in features:
            node_labels.append(features.index(node_label))

    classes = wordClasses.values()
    word_labels = [features.index(cls) for cls in classes]
    all_labels = list(node_labels) + word_labels

    # Step 5. Reading Graph and fetching its respective attributes
    textGraph = TextGraph()
    f, X, A_hat, graph = textGraph.loadGraph()

    # Step 6. Graph Details
    dataset.getGraphDetails()

    X_new = np.hstack((X, np.array(all_labels)[:, None]))
    
    X_new_train, X_new_test, A_hat_train, A_hat_test = train_test_split(X_new, A_hat, test_size=0.33, random_state=42)

    X_train = X_new_train[:, :-1]
    y_train = X_new_train[:, -1]

    X_test = X_new_test[:, :-1]
    y_test = X_new_test[:, -1]

    # Step 6. Graph Convolutional Network Model
    # model = gcn(X_train.shape[1], A_hat_train)
    model = gcn(X.shape[1], A_hat)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()
    test_loss = []
    test_accuracy = []
    test_epochs = []
    loss_per_epochs = []
    accuracy_per_epochs = []
    for epoch in range(num_of_epochs):
        model.train()
        optimizer.zero_grad()
        # output = model(X_train)
        output = model(X)
        # loss_train = loss_fun(output, torch.tensor(y_train))
        loss_train = loss_fun(output, torch.tensor(all_labels))
        loss_per_epochs.append(loss_train.item())
        # training_accuracy = accuracy(output, y_train)
        training_accuracy = accuracy(output, all_labels)
        accuracy_per_epochs.append(training_accuracy.item())
        loss_train.backward()
        optimizer.step()
        print('Epoch:' + str(epoch) + '\ttraining loss:' + str(loss_train.item()) +
              '\t training accuracy:' + str(training_accuracy.item()))
        '''if epoch%5 ==0:
            test_epochs.append(epoch)
            test_output = model(X_test)
            loss_test = loss_fun(test_output, torch.tensor(y_test))
            test_loss.append(loss_test.item())
            accuracy_test = accuracy(test_output, y_test)
            test_accuracy.append(accuracy_test.item())
            print('Epoch:' + str(epoch) + '\tTesting loss:' + str(loss_test.item()) +
                  '\t Testing accuracy:' + str(accuracy_test.item()))'''

    # save the model to disk
    pickle.dump(model, open(model_filename, 'wb'))
    plotGraph(range(num_of_epochs), loss_per_epochs, plot_x_axis, plot_y_axis_loss, training_loss_plot_file_name,
              training_loss_plot_name)
    plotGraph(range(num_of_epochs), accuracy_per_epochs, plot_x_axis, plot_y_axis_accuracy,
              training_accuracy_plot_file_name, training_accuracy_plot_name)
    '''plotGraph(test_epochs, test_loss, plot_x_axis, plot_y_axis_loss, testing_loss_plot_file_name,
              testing_loss_plot_name)
    plotGraph(test_epochs, test_accuracy, plot_x_axis, plot_y_axis_accuracy, testing_accuracy_plot_file_name,
              testing_accuracy_plot_name) '''
