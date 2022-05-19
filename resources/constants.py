input_folder = 'paper/'

output_folder = './data27/'

output_file = './output/output.txt'
dataset_details = './output/DatasetDetails.csv'
text_graph_file_name = './output/Text Graph'
text_graph_name = 'Text Graph'
training_loss_plot_file_name = './output/Training Loss per epochs'
training_loss_plot_name = 'Training Loss per epochs'
training_accuracy_plot_file_name = './output/Training Accuracy per epochs'
training_accuracy_plot_name = 'Training Accuracy per epochs'
model_filename = './output/finalized_model.sav'

testing_loss_plot_file_name = './output/Training Loss per epochs'
testing_loss_plot_name = 'Training Loss per epochs'
testing_accuracy_plot_file_name = './output/Training Accuracy per epochs'
testing_accuracy_plot_name = 'Training Accuracy per epochs'

output_column_filename = 'File Name'
output_column_noOfWords = 'Number of Words in File'
output_column_content = 'Content'

summary_column_noOfFiles = 'No Of Files in Dataset'
summary_column_noOfUniqueWords = 'No of Unique Words in Dataset'
summary_column_uniqueWords = 'Unique Words'

graph_document_nodes = 'Document Nodes'
graph_word_nodes = 'Word Nodes'

graph_no_document_nodes = 'No of document nodes'
graph_no_word_nodes = 'No of word nodes'
graph_no_nodes = 'Total No of nodes'

graph_document_edges = 'Document to word Edges'
graph_word_edges = 'Word to word Edges'

graph_no_word_edges = 'No of word edges'
graph_no_document_edges = 'No of document edges'
graph_no_edges = 'Total No of edges'

graph_details = './output/GraphDetails.csv'

text_graph_pkl_file_name = 'text_graph2.pkl'
word_edge_graph_pkl_file_name = 'word_word_edges2.pkl'

resource_path = 'resources/labels'

log_save_graph = 'Created Graph Saved...'
log_pkl_saved = 'Pkl file is already saved...'
log_add_doc_node = 'Adding document nodes to graph...'
log_building_graph = 'Building graph (No. of document, word nodes: %d, %d)...'
log_training_starts = 'Training Process starts...'

plot_x_axis = 'Epochs'
plot_y_axis_loss = 'Loss'
plot_y_axis_accuracy = 'Accuracy'

'''
No of neurons in hidden layer = (Input size * 2/3) + no of output classes
'''
hidden_layer_1_size = 330
hidden_layer_2_size = 130
no_output_classes = 7
learning_rate = 0.011
num_of_epochs = 301

