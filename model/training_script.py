import os
import logging
import argparse
import json
import yaml
import time
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from model.GNN_model import GNN
from model.layer_selector import (
    local_pooling_selection,
    global_pooling_selection,
    conv_selection,
)
import copy
from data.data import Dataset
from IPython.display import clear_output
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        dataset,
        batch_size,
        lr,
        conv_layer,
        global_pooling_layer,
        local_pooling_layer,
        attention_heads,
        hidden_channels,
        nb_max_epochs,
        patience,
        verbose,
        device,
        writer,
        split_nb,
        alpha=1e-2,
    ):
        # Creation of the dataset
        n = len(dataset)
        self.dataset_name = dataset.name
        self.split_nb = split_nb
        train_dataset = dataset[: int(0.6 * n)]
        val_dataset = dataset[int(0.6 * n): int(0.8 * n)]
        test_dataset = dataset[int(0.8 * n):]
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        # Model build
        self.device = device
        self.local_pooling_layer = local_pooling_layer
        self.convolutional_layer = conv_layer
        self.global_pooling_layer = global_pooling_layer

        local_pooling, dic_conversion_layer = local_pooling_selection(
            local_pooling_layer, device=device
        )
        convolutional_layer = conv_selection(conv_layer, attention_heads)

        self.model = GNN(
            num_node_features=dataset.num_node_features,
            num_classes=dataset.num_classes,
            hidden_channels=hidden_channels,
            conv_method=convolutional_layer,
            global_pool_method=global_pooling_selection(global_pooling_layer),
            local_pool_method=local_pooling,
            dic_conversion_layer=dic_conversion_layer,
        ).to(device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.alpha = alpha
        self.nb_max_epochs = nb_max_epochs
        self.patience = patience

        self.verbose = verbose
        self.writer = writer

    def train(self):
        self.model.train()
        for (
            data
        ) in (
            self.train_loader
        ):  # Iterate in batches over the training dataset.
            data = data.to(self.device)
            out, losses = self.model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
            loss = self.criterion(out, data.y) + self.alpha * torch.sum(
                losses
            )  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def test(self, model, loader, device):
        model.eval()
        loss_epoch = []
        correct = 0
        for (
            data
        ) in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out, losses = model(data.x, data.edge_index, data.batch)
            loss = self.criterion(out, data.y) + self.alpha * torch.sum(
                losses
            )  # Compute the loss.
            loss_epoch.append(loss.detach().cpu().item())
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int(
                (pred == data.y).sum()
            )  # Check against ground-truth labels.
        return correct / len(loader.dataset), np.mean(
            loss_epoch
        )  # Derive ratio of correct predictions.

    def training_loop(self):
        start = time.time()

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_acc = 0
        iterations_WO_improvements = 0

        with tqdm(range(1, self.nb_max_epochs), unit="epoch") as bar:
            for epoch in range(1, self.nb_max_epochs):
                bar.set_description((
                    f"{self.dataset_name}"
                    f"{self.convolutional_layer}"
                    f"{self.local_pooling_layer}"
                    f"{self.global_pooling_layer}"
                    f"- Split {self.split_nb}"
                    f"- Epoch {epoch}")
                )
                self.train()
                train_acc, train_loss = self.test(
                    self.model, self.train_loader, self.device
                )
                val_acc, val_loss = self.test(
                    self.model, self.val_loader, self.device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                bar.set_postfix(train_acc=train_acc, val_acc=val_acc)
                bar.update(1)
                clear_output(wait=False)

                # Tensorboard dashboard
                self.writer.add_scalar((
                    f"{self.dataset_name}"
                    f"/split_{self.split_nb}"
                    f"/training_loss"),
                    train_loss,
                    epoch,
                )

                self.writer.add_scalar((
                    f"{self.dataset_name}"
                    f"/split_{self.split_nb}"
                    f"/validation_loss"),
                    val_loss,
                    epoch,
                )

                self.writer.add_scalar((
                    f"{self.dataset_name}"
                    f"/split_{self.split_nb}"
                    f"/training_accuracy"),
                    train_acc,
                    epoch,
                )

                self.writer.add_scalar((
                    f"{self.dataset_name}"
                    f"/split_{self.split_nb}"
                    f"/validation_accuracy"),
                    val_acc,
                    epoch,
                )

                # Early stopping
                if val_acc >= best_acc:
                    best_acc = val_acc
                    min_val_loss = val_loss
                    iterations_WO_improvements = 0
                    best_model = copy.deepcopy(self.model)
                else:
                    iterations_WO_improvements += 1

                logger.info(
                    f"Epoch: {epoch:03d}"
                    f", Train Acc: {train_acc:.4f}"
                    f", Val Acc: {val_acc:.4f}"
                )

                if iterations_WO_improvements > self.patience:
                    break

        test_acc, _ = self.test(best_model, self.test_loader, self.device)
        last_epoch = epoch
        stop = time.time()
        return (
            best_model,
            test_acc,
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            last_epoch,
            min_val_loss,
            best_acc,
            (stop - start) / last_epoch,
        )


def train_model_from_config(file_path):
    # Reading the config file
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)["model"]

    output_model_path = config.pop("output_model_path", "model/weights")
    output_results_path = config.pop("output_results_path", "model/results")
    device = config.pop("device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verbose = config.pop("verbose", 2)
    dataset_path = config.pop("dataset_path", "data/datasets")
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

    test_accuracy_list = []

    logger.info("\n" + dataset_name)
    logger.info("Convolutional layer :" + conv_layer)
    logger.info("Pooling layer : " + global_pooling_layer)
    logger.info("Readout layer : " + str(local_pooling_layer))

    writer = SummaryWriter(
        f"logs/runs/{conv_layer}_{global_pooling_layer}_{local_pooling_layer}"
    )

    for i in range(nb_of_splits):
        torch.manual_seed(12345 + i)
        torch.cuda.manual_seed_all(12345 + i)
        dataset = dataset.shuffle()
        trainer = Trainer(
            dataset,
            batch_size,
            lr,
            conv_layer,
            global_pooling_layer,
            local_pooling_layer,
            attention_heads,
            hidden_channels,
            max_epochs,
            patience,
            verbose,
            device,
            writer,
            i + 1,
            alpha,
        )

        # Model training
        (
            best_model,
            test_acc,
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            last_epoch,
            min_val_loss,
            best_acc,
            train_time,
        ) = trainer.training_loop()
        result["split " + str(i + 1)] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "test_accuracy": test_acc,
            "last_epoch": last_epoch,
            "train_time_per_epoch": train_time,
        }
        logger.info((
            f"Model number: {i:02d}"
            f", Train acc: {train_accuracies[-1]:.4f}"
            f", Test Acc: {test_acc:.4f}"
            f", stopped at epoch {last_epoch}"
            f"  -> best val loss: {min_val_loss:.4f}"
            f" , best val acc: {best_acc:.4f}"
        ))
        test_accuracy_list.append(test_acc)

    # Compute accuracies and informations about the model
    result["nb_parameters"] = sum(
        p.numel() for p in best_model.parameters() if p.requires_grad
    )
    result["mean_accuracy"] = np.mean(test_accuracy_list)
    result["std_accuracy"] = np.std(test_accuracy_list)

    # Model saving
    logger.info((
        f"Mean Test Acc: {result['mean_accuracy']:.4f}"
        f", Std Test Acc: {result['std_accuracy']:.4f}")
    )

    os.makedirs(output_model_path, exist_ok=True)

    model_name = (f"{dataset_name}"
                  f"_{conv_layer}"
                  f"_{global_pooling_layer}"
                  f"_{local_pooling_layer}")

    torch.save(
        best_model.state_dict(), os.path.join(output_model_path, model_name)
    )

    os.makedirs(output_results_path, exist_ok=True)

    with open(
        os.path.join(output_results_path, model_name) + ".json", "w"
    ) as json_file:
        json.dump(result, json_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a YAML configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/config_test.yml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    file_path = args.config
    train_model_from_config(file_path)
