import os
import torch
import numpy as np
import networkx as nx
import torch_geometric
from torch_geometric.utils import homophily
import csv
from torch_geometric import datasets

def get_homophily(name_location,name_dataset,seed_id=12345):
    # Name of the CSV file
    csv_file = "data/homophily/homophily_data.csv"
    
    # Check if the file exists
    file_exists = os.path.exists(csv_file)
    
    # Open CSV file in append mode if it exists, otherwise in write mode
    mode = 'a' if file_exists else 'w'
    
    dataset = eval("datasets."+name_location)(os.path.join("data", name_location), name=name_dataset)

    size_dataset = len(dataset)
    nb_class = dataset.num_classes
    nb_features = dataset.num_features

    torch.manual_seed(seed_id)
    dataset = dataset.shuffle()
    length_train_dataset = int(np.ceil(0.8*len(dataset)))

    train_dataset = dataset[:length_train_dataset]
    test_dataset = dataset[length_train_dataset:]
    
    homophily_edge_train = round(homophily(train_dataset.edge_index, torch.argmax(train_dataset.x, dim=1), method='edge'),3)
    homophily_node_train = round(homophily(train_dataset.edge_index, torch.argmax(train_dataset.x, dim=1), method='node'),3)
    homophily_edge_insensitive_train = round(homophily(train_dataset.edge_index, torch.argmax(train_dataset.x, dim=1), method='edge_insensitive'),3)
    
    homophily_edge_test = round(homophily(test_dataset.edge_index, torch.argmax(test_dataset.x, dim=1), method='edge'),3)
    homophily_node_test = round(homophily(test_dataset.edge_index, torch.argmax(test_dataset.x, dim=1), method='node'),3)
    homophily_edge_insensitive_test = round(homophily(test_dataset.edge_index, torch.argmax(test_dataset.x, dim=1), method='edge_insensitive'),3)
    
    line_csv = [
    {"Name_Dataset": name_dataset, "Size_dataset": size_dataset, 
     "Nb_class": nb_class, "Nb_features": nb_features, "Seed": seed_id,
     "Homophily_edge_train": homophily_edge_train, "Homophily_edge_test": homophily_edge_test,
     "Homophily_node_train": homophily_node_train, "Homophily_node_test": homophily_node_test,
     "Homophily_edge_insensitive_train": homophily_edge_insensitive_train, "Homophily_edge_insensitive_test": homophily_edge_insensitive_test}
        ]
    
    # Writing to CSV file
    with open(csv_file, mode, newline='') as file:
        # Define column names
        fieldnames = ["Name_Dataset", "Size_dataset","Nb_class","Nb_features", "Seed",
                      "Homophily_edge_train", "Homophily_edge_test",
                      "Homophily_node_train", "Homophily_node_test",
                      "Homophily_edge_insensitive_train", "Homophily_edge_insensitive_test"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header only if file is created newly
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writeheader()


        # Write data rows
        for row in line_csv:
            writer.writerow(row)

    print("Name of the dataset: " + name_dataset)
    print("Size of the dataset: " + str(size_dataset))
    print("Number of features: " + str(nb_features))
    print("Number of classes: " + str(nb_class))
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print("Homophily with the edge formula (train/test): " + str(homophily_edge_train) + " | " + str(homophily_edge_test))
    print("Homophily with the node formula (train/test): " + str(homophily_node_train) + " | " + str(homophily_node_test))
    print("Homophily with the edge_insensitive formula (train/test): " + str(homophily_edge_insensitive_train) + " | " + str(homophily_edge_insensitive_test))
    print("CSV file created successfully:", csv_file)