from __future__ import division
from __future__ import print_function

from datasetlaoding import Dataset
import torch
from textGraph import TextGraph
from gcnmodel import gcn
import torch.optim as optim
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def accuracy(output, labels):
    _, prediction = output.max(1)
    prediction = prediction.numpy()
    actual_labels = [(label) for label in labels]
    correct = sum(actual_labels == prediction)
    return correct / len(prediction)

def plotGraph(x, y, x_label, y_label, graph_title):
    plt.figure(figsize=(15, 4))
    plt.title(graph_title, fontdict={'fontweight': 'bold', 'fontsize': 18})
    plt.plot(x, y, label= graph_title)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(graph_title+'png', dpi=100)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = Dataset()
    # dataset.networkxToDglgraph()

    features = dataset.generateLabels()
    wordClasses, index_doc = dataset.readFiles(features)
    # print(wordClasses)


    #one_hot_encoding_labels = dataset.encodeLabeles(classes, index_doc)
    # print(one_hot_encoding_labels)

    textGraph = TextGraph()
    f, X, A_hat = textGraph.loadGraph()

    node_labels_values = list(index_doc.values())

    node_labels = []
    for node_label in node_labels_values:
        if node_label in features:
            node_labels.append(features.index(node_label))

    classes = wordClasses.values()
    word_labels = [features.index(cls) for cls in classes]
    all_labels = list(node_labels) + word_labels
    model = gcn(X.shape[1], A_hat)
    optimizer = optim.Adam(model.parameters())
    loss_fun = torch.nn.CrossEntropyLoss()

    loss_per_epochs = []
    accuracy_per_epochs = []
    print('Training Process starts...')
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

    plotGraph(range(201),loss_per_epochs,'Epochs','Loss','Loss per epochs')
    plotGraph(range(201),accuracy_per_epochs, 'Epochs', 'Accuracy', 'Accuracy per epochs')
