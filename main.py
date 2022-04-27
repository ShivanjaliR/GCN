'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''

from __future__ import division
from __future__ import print_function

import sys
from datasetlaoding import Dataset
import torch
from resources.constants import accuracy_plot_name, loss_plot_name, output_file, loss_plot_file_name, \
    accuracy_plot_file_name, log_training_starts, plot_x_axis, plot_y_axis_loss, plot_y_axis_accuracy
from textGraph import TextGraph
from gcnmodel import gcn
import torch.optim as optim
from utils import plotGraph, accuracy, generateLabels

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
    f, X, A_hat = textGraph.loadGraph()

    # Step 6. Graph Convolutional Network Model
    model = gcn(X.shape[1], A_hat)
    optimizer = optim.Adam(model.parameters())
    loss_fun = torch.nn.CrossEntropyLoss()

    loss_per_epochs = []
    accuracy_per_epochs = []
    print(log_training_starts)
    for epoch in range(201):
        model.train()
        optimizer.zero_grad()
        output = model(f)
        loss_train = loss_fun(output, torch.tensor(all_labels))
        loss_per_epochs.append(loss_train.item())
        training_accuracy = accuracy(output, all_labels)
        accuracy_per_epochs.append(training_accuracy.item())
        loss_train.backward()
        optimizer.step()
        print('Epoch:' + str(epoch) + '\ttraining loss:'+ str(loss_train.item()) +
          '\t training accuracy:'+ str(training_accuracy.item()))

    plotGraph(range(201),loss_per_epochs,plot_x_axis,plot_y_axis_loss, loss_plot_file_name, loss_plot_name)
    plotGraph(range(201),accuracy_per_epochs, plot_x_axis, plot_y_axis_accuracy, accuracy_plot_file_name, accuracy_plot_name)
