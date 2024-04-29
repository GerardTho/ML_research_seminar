import torch
from torch_geometric.loader import DataLoader
from model.GNN_model import GCN
from model.layer_selector import local_pooling_selection, global_pooling_selection, conv_selection
import torch_geometric.nn as nn
import copy
from torch_geometric import datasets
from data.data import Dataset
import os
import numpy as np
import json
import yaml
import argparse
import time
from tqdm import tqdm
from IPython.display import clear_output

class Trainer():
  
  def __init__(self, dataset, batch_size, lr, conv_layer, global_pooling_layer, local_pooling_layer, attention_heads, hidden_channels, nb_max_epochs, patience, verbose, device, alpha=1e-2):
    # Creation of the dataset
    n = len(dataset)
    train_dataset = dataset[:int(0.6*n)]
    val_dataset = dataset[int(0.6*n):int(0.8*n)]
    test_dataset = dataset[int(0.8*n):]
    self.batch_size = batch_size
    self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model build
    self.device = device
    local_pooling, dic_conversion_layer = local_pooling_selection(local_pooling_layer, device=device)
    convolutional_layer=conv_selection(conv_layer, attention_heads)

    self.model = GCN(num_node_features=dataset.num_node_features, 
                num_classes=dataset.num_classes, 
                hidden_channels=hidden_channels,
                conv_method=convolutional_layer, 
                global_pool_method=global_pooling_selection(global_pooling_layer), 
                local_pool_method=local_pooling,
                dic_conversion_layer=dic_conversion_layer).to(device)
    self.lr = lr
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
    self.criterion = torch.nn.CrossEntropyLoss()

    self.alpha = alpha
    self.nb_max_epochs = nb_max_epochs
    self.patience = patience
    self.verbose = verbose

  def train(self):
      self.model.train()
      for data in self.train_loader:  # Iterate in batches over the training dataset.
          data=data.to(self.device)
          out, losses = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
          loss = self.criterion(out, data.y) + self.alpha*torch.sum(losses)  # Compute the loss.
          loss.backward()  # Derive gradients.
          self.optimizer.step()  # Update parameters based on gradients.
          self.optimizer.zero_grad()  # Clear gradients.

  def test(self, model, loader, device):
      model.eval()
      loss_epoch = []
      correct = 0
      for data in loader:  # Iterate in batches over the training/test dataset.
          data=data.to(device)
          out, losses = model(data.x, data.edge_index, data.batch)
          loss = self.criterion(out, data.y) + self.alpha*torch.sum(losses)  # Compute the loss.
          loss_epoch.append(loss.detach().cpu().item())
          pred = out.argmax(dim=1)  # Use the class with highest probability.
          correct += int((pred == data.y).sum())  # Check against ground-truth labels.
      return correct / len(loader.dataset), np.mean(loss_epoch)  # Derive ratio of correct predictions.

  def training_loop(self):
    start = time.time()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_acc = 0
    min_val_loos = np.inf
    iterations_WO_improvements = 0
    with tqdm(range(1, self.nb_max_epochs), unit='epoch') as bar:
      for epoch in range(1, self.nb_max_epochs):
        bar.set_description(f'Epoch {epoch}')
        self.train()
        train_acc, train_loss = self.test(self.model, self.train_loader, self.device)
        val_acc, val_loss = self.test(self.model, self.val_loader, self.device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        bar.set_postfix(train_acc=train_acc, val_acc=val_acc)
        bar.update(1)
        clear_output(wait=False)

        # Early stopping
        if val_acc >= best_acc:
          best_acc = val_acc
          min_val_loss = val_loss
          iterations_WO_improvements = 0
          best_model = copy.deepcopy(self.model)
        else:
          iterations_WO_improvements += 1

        if iterations_WO_improvements > self.patience:
          break

        if self.verbose>1:
          # Print should be replaced by logs ideally
          print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    test_acc, _ = self.test(best_model, self.test_loader, self.device)
    last_epoch = epoch
    stop = time.time()
    return best_model, test_acc, train_losses, val_losses, train_accuracies, val_accuracies, last_epoch, min_val_loss, best_acc, (stop-start)/last_epoch


def train_model_from_config(file_path):
 # Reading the config file
  with open(file_path, 'r') as file:
      config = yaml.safe_load(file)["model"]

  output_model_path = config.pop("output_model_path", "model/weights")
  output_results_path = config.pop("output_results_path", "model/results")
  device = config.pop("device", None)
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  verbose = config.pop("verbose", 2)
  dataset_path = config.pop("dataset_path", "data/datasets")
  location = dataset_path.split("/")[-1]
  nb_of_splits = config.pop("nb_of_splits", 10)
  hidden_channels = config.get("hidden_channels", 32)

  result = dict(config)
  dataset_name = config.pop("dataset")
  max_epochs = config.pop("max_epochs", 200)
  patience = config.pop("patience", 20)
  lr = config.pop("lr", 0.005)
  alpha = float(config.pop("alpha", 1e-2))
  batch_size = config.pop("batch_size", 64)
  conv_layer = config.pop("convolution_layer", "GCN")
  attention_heads = config.pop("attention_heads", 4)
  global_pooling_layer = config.pop("global_pooling_layer", "mean")
  local_pooling_layer = config.pop("local_pooling_layer", "SAG")
    
  use_deterministic_algorithms = config.pop("deterministic_algorithms", True)
  if use_deterministic_algorithms:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

  # Loading the dataset with the string
  dataset = Dataset(dataset_path, name=dataset_name)

  best_test_acc = 0
  test_accuracy_list = [] 
    
  print('\n' + str(dataset_name))
  print(conv_layer, global_pooling_layer, local_pooling_layer)
    
  for i in range(nb_of_splits):
    torch.manual_seed(12345+i)
    torch.cuda.manual_seed_all(12345+i)
    dataset = dataset.shuffle()

    trainer = Trainer(dataset, batch_size, lr, conv_layer, global_pooling_layer, local_pooling_layer, attention_heads, hidden_channels, max_epochs, patience, verbose, device, alpha)
    
    # Model training
    best_model, test_acc, train_losses, val_losses, train_accuracies, val_accuracies, last_epoch, min_val_loss, best_acc, train_time = trainer.training_loop()
    result["split "+str(i+1)] = {"train_losses":train_losses,
                               "val_losses":val_losses,
                               "train_accuracies":train_accuracies,
                               "val_accuracies":val_accuracies,
                               "test_accuracy":test_acc,
                               "last_epoch":last_epoch, 
                               "train_time_per_epoch": train_time}
    if verbose > 0:
      print(f'Model number: {i:02d}, Train acc: {train_accuracies[-1]:.4f}, Test Acc: {test_acc:.4f}, stopped at epoch {last_epoch} -> best val loss: {min_val_loss:.4f}, best val acc: {best_acc:.4f}')
    test_accuracy_list.append(test_acc)

  # Compute accuracies and informations about the model
  result["nb_parameters"] = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
  result["mean_accuracy"] = np.mean(test_accuracy_list)
  result["std_accuracy"] = np.std(test_accuracy_list)

  # Model saving
  print(f'Mean Test Acc: {result["mean_accuracy"]:.4f}, Std Test Acc: {result["std_accuracy"]:.4f}')
  
  os.makedirs(output_model_path, exist_ok = True) 

  model_name = f"{dataset_name}_{conv_layer}_{global_pooling_layer}_{local_pooling_layer}"

  torch.save(best_model.state_dict(), os.path.join(output_model_path, model_name))

  os.makedirs(output_results_path, exist_ok = True)

  with open(os.path.join(output_results_path, model_name) + ".json", 'w') as json_file:
      json.dump(result, json_file, indent=2)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process a YAML configuration file and output the results to a JSON file.')
  parser.add_argument('-c', '--config', default="configs\config_test.yml", help='Path to the YAML configuration file')
  args = parser.parse_args()

  file_path=args.config
  train_model_from_config(file_path)