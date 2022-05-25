'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''

from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from datasetlaoding import Dataset
import torch
from resources.constants import training_accuracy_plot_name, training_loss_plot_name, output_file, \
    training_loss_plot_file_name, \
    training_accuracy_plot_file_name, plot_x_axis, plot_y_axis_loss, plot_y_axis_accuracy, learning_rate, \
    num_of_epochs, model_filename, testing_accuracy_plot_file_name, testing_accuracy_plot_name, \
    testing_loss_plot_file_name, testing_loss_plot_name, test_index_file_name, selected_index_file, not_selected_file, \
    selected_label_file, not_selected_label_file, training_dataset_size, testing_dataset_size, sliding_window_size
from textGraph import TextGraph
from gcnmodel import gcn
import torch.optim as optim
from utils import plotGraph, accuracy, generateLabels, save_as_pickle
import pickle

if __name__ == '__main__':

    sys.stdout = open(output_file, "w")

    # Step 1: Dataset Generation and Cleaning
    dataset = Dataset()
    features = generateLabels()
    dataset.readFilesDocCleaning(features)
    index_doc = dataset.getIndexDoc()

    # Step 2: Frequency Calculation
    dataset.FrequencyCalculation(sliding_window_size)

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

    # Step 5: Split Training and Testing Dataset
    test_idxs = []
    test_ratio = .20
    for cls in range(len(features)):
        dum = [index for index, c in enumerate(all_labels) if cls == c]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum, size=round(test_ratio * len(dum)), replace=False)))

    save_as_pickle(test_index_file_name, test_idxs)
    # select only certain labelled nodes for semi-supervised GCN
    selected = []
    not_selected = []
    for i in range(len(all_labels)):
        if i not in test_idxs:
            selected.append(i)
        else:
            not_selected.append(i)
    save_as_pickle(selected_index_file, selected)
    save_as_pickle(not_selected_file, not_selected)

    labels_selected = [l for idx, l in enumerate(all_labels) if idx in selected]
    labels_not_selected = [l for idx, l in enumerate(all_labels) if idx not in selected]
    save_as_pickle(selected_label_file, labels_selected)
    save_as_pickle(not_selected_label_file, labels_not_selected)

    print(training_dataset_size, len(labels_selected))
    print(testing_dataset_size, len(labels_not_selected))

    # Step 6. Reading Graph and fetching its respective attributes
    textGraph = TextGraph()
    f, X, A_hat, graph = textGraph.loadGraph()

    # Step 7. Graph Details
    dataset.getGraphDetails()

    # Step 8. Graph Convolutional Network Model
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
        output = model(X)
        loss_train = loss_fun(output[selected], torch.tensor(labels_selected))
        loss_per_epochs.append(loss_train.item())
        training_accuracy = accuracy(output[selected], labels_selected)
        accuracy_per_epochs.append(training_accuracy.item())
        loss_train.backward()
        optimizer.step()
        print('Epoch:' + str(epoch) + '\ttraining loss:' + str(loss_train.item()) +
              '\t training accuracy:' + str(training_accuracy.item()))
        if epoch % 5 == 0:
            test_epochs.append(epoch)
            test_output = model(X)
            loss_test = loss_fun(test_output[not_selected], torch.tensor(labels_not_selected))
            test_loss.append(loss_test.item())
            accuracy_test = accuracy(test_output[not_selected], labels_not_selected)
            test_accuracy.append(accuracy_test.item())
            print('Epoch:' + str(epoch) + '\tTesting loss:' + str(loss_test.item()) +
                  '\t Testing accuracy:' + str(accuracy_test.item()))

    # save the model to disk
    pickle.dump(model, open(model_filename, 'wb'))
    plotGraph(range(num_of_epochs), loss_per_epochs, plot_x_axis, plot_y_axis_loss, training_loss_plot_file_name,
              training_loss_plot_name)
    plotGraph(range(num_of_epochs), accuracy_per_epochs, plot_x_axis, plot_y_axis_accuracy,
              training_accuracy_plot_file_name, training_accuracy_plot_name)
    plotGraph(test_epochs, test_loss, plot_x_axis, plot_y_axis_loss, testing_loss_plot_file_name,
              testing_loss_plot_name)
    plotGraph(test_epochs, test_accuracy, plot_x_axis, plot_y_axis_accuracy, testing_accuracy_plot_file_name,
              testing_accuracy_plot_name)
