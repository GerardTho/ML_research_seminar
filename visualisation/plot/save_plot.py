from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

import os


class Plot :

    def __init__(self, list_dict : List[Dict]) -> None :

        self.list_dict = list_dict
        self.datasets = set(d["dataset"] for d in list_dict)
        self.set_sorted_list_dict()
        self.set_worst_best_acc_per_dataset()

    def save_fig_in_folder(self, path) -> None:
        directory, _ = os.path.split(path)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(path)

    def set_sorted_list_dict(self) -> None :
        """
        Sort the list of dictionaries in alphabetical order with respect to
        the architecture, the pooling and the dataset
        """
        list_dict_architecture_as_key = {
            f"{d['convolution_layer']}_{d['local_pooling_layer']}_{d['dataset']}": d
            for d in self.list_dict
        }
        list_dict_architecture_as_key = dict(
            sorted(list_dict_architecture_as_key.items())
        )
        self.sorted_list_dict = list(list_dict_architecture_as_key.values())


    def set_worst_best_acc_per_dataset(self) -> None :
        """
        Create new attributes corresponding to the best
        and the worst accuracy for each dataset
        
        These attributes are dictionary whose keys are the
        datasets and values are the mean accuracy and the whole
        corresponing dictionary (a tuple)
        """
        best_accs = {}
        worst_accs = {}

        for dic in self.sorted_list_dict:
            dataset = dic.get("dataset")
            mean = dic.get("mean_accuracy")

            if dataset not in best_accs.keys():
                best_accs[dataset] = (mean, dic)
                worst_accs[dataset] = (mean, dic)
            else:
                best_mean_dataset = best_accs[dataset][0]

                if mean > best_mean_dataset:
                    best_accs[dataset] = (mean, dic)

                worst_mean_dataset = worst_accs[dataset][0]

                if mean < worst_mean_dataset:
                    worst_accs[dataset] = (mean, dic)

        self.worst_accs = worst_accs
        self.best_accs = best_accs

 
    def plot_losses(self, train : bool = True) -> None :
        """
        Plot the losses as a function of the epochs of the train/validation

        train -> if True, plot the train, else plot the validation
        """
        kind = "train" if train else "val"

        for dic in self.list_dict:
            
            # if the dictionary is valid
            if "split 1" in dic:

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)

                ax.plot(
                    np.arange(len(dic["split 1"][f"{kind}_losses"])),
                    dic["split 1"][f"{kind}_losses"],
                )

                ax.set_xlabel("Epochs")
                ax.set_ylabel(f"{kind} loss")

                ax.set_title(
                    f"{kind} loss across epochs with early stopping \n for " \
                    f"{dic['dataset']} with {dic['convolution_layer']} and " \
                    f"{dic['local_pooling_layer']}"
                )

                    
                self.save_fig_in_folder(
                os.path.join("visualisation",
                            "results",
                             "losses",
                             kind, 
                             f"{dic['dataset']}_{dic['convolution_layer']}_" \
                             f"{dic['local_pooling_layer']}.png"
                    )
                )
                
                plt.close()


    def plot_acc(self, train : bool = True) -> None :
        """
        Plot the losses as a function of the epochs of the train/validation

        train -> if True, plot the train, else plot the validation
        """
        kind = "train" if train else "val"

        for dic in self.list_dict:
            
            # if the dictionary is valid
            if "split 1" in dic:

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)

                ax.plot(
                    np.arange(len(self.list_dict[0]["split 1"][f"{kind}_accuracies"])),
                    self.list_dict[0]["split 1"][f"{kind}_accuracies"],
                )

                ax.set_xlabel("Epochs")
                ax.set_ylabel(f"{kind} accuracy")

                ax.set_title(
                    f"{kind} accuracy across epochs with early stopping \n for " \
                    f"{dic['dataset']} with {dic['convolution_layer']} and " \
                    f"{dic['local_pooling_layer']}"
                    )
                    
                self.save_fig_in_folder(
                os.path.join("visualisation",
                            "results",
                             "acc",
                             kind, 
                             f"{dic['dataset']}_{dic['convolution_layer']}_" \
                             f"{dic['local_pooling_layer']}.png"
                    )
                )
                
                plt.close()


    def plot_acc_and_loss(self, train : bool = True) -> None :

        """
        Plot the losses and the accuracy as a function of 
        the epochs of the train/validation

        train -> if True, plot the train, else plot the validation
        """
        kind = "train" if train else "val"

        for dic in self.list_dict:
            
            # if the dictionary is valid
            if "split 1" in dic:

                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)

                color1 = "tab:blue"

                ax1.plot(
                    np.arange(len(self.list_dict[0]["split 1"][f"{kind}_accuracies"])),
                    self.list_dict[0]["split 1"][f"{kind}_accuracies"],
                    color=color1,
                    label= f"{kind} accuracy"
                )

                ax1.set_xlabel("Epochs")
                ax1.set_ylabel(f"{kind} accuracy", color=color1)
                ax1.tick_params(axis="y", labelcolor=color1)

                
                # Creating a secondary y-axis for the second curve
                ax2 = ax1.twinx()
                color2 = "tab:red"

                ax2.plot(
                    np.arange(len(self.list_dict[0]["split 1"][f"{kind}_losses"])),
                    self.list_dict[0]["split 1"][f"{kind}_losses"],
                    color=color2,
                    label= f"{kind} loss"
                )

                ax2.set_xlabel("Epochs")
                ax1.set_ylabel(f"{kind} loss", color=color2)
                ax2.tick_params(axis="y", labelcolor=color2)

                # Adding legends
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2, loc="upper left")

                ax1.set_title(
                    f"{kind} accuracy and loss across epochs with early stopping \n for " \
                    f"{dic['dataset']} with {dic['convolution_layer']} and " \
                    f"{dic['local_pooling_layer']}"
                    )

                plt.tight_layout()

                self.save_fig_in_folder(
                os.path.join("visualisation",
                            "results",
                             "acc_and_loss",
                             kind, 
                             f"{dic['dataset']}_{dic['convolution_layer']}_" \
                             f"{dic['local_pooling_layer']}.png"
                    )
                )

                plt.close()

    def plot_all(self, train : bool = True) -> None :
        """
        Plot everything (losses, accuracy, and losses and accuracy) 
        as a function of the epochs of the train/validation

        train -> if True, plot the train, else plot the validation
        """
        self.plot_losses(train)
        self.plot_acc(train)
        self.plot_acc_and_loss(train)