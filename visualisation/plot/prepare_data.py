from typing import List, Dict

from json import load
import os
from mimetypes import guess_type

from statistics import median
import pandas as pd


class Prepare:

    def __init__(self):

        self._path_list_dict = os.path.join("model", "results")
        self._path_homophily = os.path.join(
            "data/homophily", "homophily_data.csv"
        )

        # taken from https://chrsmrrs.github.io/datasets/docs/datasets/
        # for each dataset, the tuple of the average number of nodes and
        # edges respectively
        self.additional_information = {
            "ENZYMES": (32.63, 62.14),
            "PROTEINS": (39.06, 72.82),
            "MUTAG": (17.93, 19.79),
            "NCI1": (29.87, 32.30),
        }

        self.set_list_dict()
        self.drop_mean_pooling()
        self.set_median_training_time()
        self.set_additional_information()

    def set_list_dict(self) -> None:
        """
        Return the list of dictionary that is used for most
        visualisation task
        """
        list_dict = []

        with os.scandir(self._path_list_dict) as entries:
            for entry in entries:
                if entry.is_file():
                    name = entry.name
                    if "json" in guess_type(name)[0]:
                        # Construct the full path using os.path.join again
                        full_path = os.path.join(self._path_list_dict, name)
                        with open(full_path, "r") as file:
                            data = load(file)
                        list_dict.append(data)

        self.list_dict = list_dict

    def drop_mean_pooling(self) -> None:
        """
        Drop all the entries that are not mean
        """
        self.list_dict = [
            dic
            for dic in self.list_dict
            if dic["global_pooling_layer"] != "mean"
        ]

    def set_median_training_time(self) -> None:
        """
        Add a new key corresponding to the median of the training
        time along the splits for all dictionary of list_dict
        """
        for dic in self.list_dict:
            res = []

            for key, value in dic.items():
                if "split" in key:
                    res.append(
                        value["last_epoch"] * value["train_time_per_epoch"]
                    )

            dic["Training time"] = median(res)

    def set_additional_information(self) -> None:
        """
        Add extra information on the dictionaries of list_dict:
        the homophily of the dataset, the average number of nodes
        and edges, and a new variable beeing the pooling and the
        architecture
        """
        df_homophily = pd.read_csv(self._path_homophily)
        df_homophily["Name_Dataset"] = df_homophily["Name_Dataset"].apply(
            lambda s: s.upper()
        )

        for dic in self.list_dict:

            name_dataset = dic["dataset"]

            dic["homophily"] = df_homophily.loc[
                df_homophily["Name_Dataset"] == name_dataset,
                "Homophily_edge_train",
            ].values[0]

            if dic["local_pooling_layer"] is None:
                dic["local_pooling_layer"] = "None"

            dic["pooling_and_archi"] = (
                f"{dic['local_pooling_layer']}+{dic['convolution_layer']}"
            )

            avg_nodes, avg_edges = self.additional_information[name_dataset]
            dic["avg_nodes"] = avg_nodes
            dic["avg_edges"] = avg_edges

    def get_list_dict(self) -> List[Dict]:
        return self.list_dict


def get_list_dict() -> List[Dict]:
    prepare = Prepare()
    return prepare.get_list_dict()
