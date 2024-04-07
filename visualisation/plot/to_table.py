from typing import List, Dict, Tuple
import pandas as pd

import os


class ToTable() :

    def __init__(self, list_dict : List[Dict], per_dataset : bool = True, global_pooling : str = "max") -> None :
        """
        Per dataset indicates whether the aggregator used for the best and
        worst results (four dataframes in total) is per datasets or not 
        (aggregate on all datasets)
        global_pooling indicates whether the computation is done over all 
        the models or just the models using mean global pooling or max
        global pooling. Put nothing in order to compute over everything.
        """
        self.list_dict = list_dict        
        self.per_dataset = per_dataset
        self.global_pooling = global_pooling
        if self.global_pooling == "":
            self.level = [0,1,2]
            self.level_transpose = [1,2,0]
        else:
            self.level = [0,1]
            self.level_transpose = [1,0]
        self.save_path = os.path.join('visualisation','results', 'tables')
        self.save_file_name_df = "full_result.txt"
        self.save_file_name_average_ranking_df_archi = 'average_ranking_archi.txt'
        self.save_file_name_average_ranking_df_pooling = 'average_ranking_pooling.txt'
        self.save_file_name_df_by_architecture = 'result_by_archi.txt'
        self.save_file_name_df_by_pooling = 'result_by_pooling.txt'
        self.save_file_name_df_best_architecture = 'best_architecture.txt'
        self.save_file_name_df_worst_architecture = 'worst_architecture.txt'
        self.save_file_name_df_best_pooling = 'best_pooling.txt'
        self.save_file_name_df_worst_pooling = 'worst_pooling.txt'

        self.set_transpose()
        if self.global_pooling != "":
            self.filter()
        self.to_bold()
        self.rename_columns()
        self.set_datasets()
        self.pivot()
        self.set_ranking()
        self.set_conv_and_pool_map()
        self.set_aggregator()
        self.set_best()
        self.set_worst()
        self.modify_training_time()
        self.set_average_ranking()
        self.set_df_by()

    # see if it is possible to delete this function
    def set_transpose(self) -> None :
        """
        Set the attribute dic_results being the transposition
        of list_dict (meaning instead of having a list of dictionary
        of values, have a dictionary of list of values)

        The new attribute will be 'df'
        """
        dic_results = {
            "dataset": [],
            "global_pooling_layer": [],
            "local_pooling_layer": [],
            "convolution_layer": [],
            "mean_accuracy": [],
            "std_accuracy": [],
            "training_time": [],
        }

        for dic in self.list_dict:
            dic_results["dataset"].append(dic["dataset"])
            dic_results["global_pooling_layer"].append(dic["global_pooling_layer"])
            dic_results["local_pooling_layer"].append(dic["local_pooling_layer"])
            dic_results["convolution_layer"].append(dic["convolution_layer"])
            dic_results["mean_accuracy"].append(dic["mean_accuracy"])
            dic_results["std_accuracy"].append(dic["std_accuracy"])
            training_time = 0

            n_error = 0
            for i in range(1, 11):
                try:
                    training_time += (
                        dic["split " + str(i)]["train_time_per_epoch"]
                        * dic["split " + str(i)]["last_epoch"]
                    )
                except KeyError:
                    training_time += 0
                    n_error += 1

            dic_results["training_time"].append(training_time / (10 - n_error))

        self.df = pd.DataFrame(dic_results)

    def filter(self) -> None :
        self.df = self.df[self.df["global_pooling_layer"]==self.global_pooling]

    def to_bold(self) -> None :
        """
        Bold the values of the dataframe (self.df) that corresponds to the maximum
        """
        indexes_to_bold = self.df.groupby("dataset")["mean_accuracy"].idxmax()

        self.df['mean_accuracy'] = self.df['mean_accuracy'].apply('{:.3f}'.format).astype(str)
        self.df['std_accuracy'] = self.df['std_accuracy'].apply('{:.3f}'.format).astype(str)

        self.df['accuracy'] = \
            self.df.apply(lambda row: '${} \\pm {}$'.format(row['mean_accuracy'], row['std_accuracy']), axis=1)

        self.df.loc[indexes_to_bold, 'accuracy'] = \
            self.df.loc[indexes_to_bold].apply(lambda row: '$\\bm{{{} \\pm {}}}$'.format(row['mean_accuracy'], row['std_accuracy']), axis=1)

    def rename_columns(self) -> None :
        """
        Rename the columns to reduce the size of the column of the future latex table
        """
        self.df = self.df.rename(
            columns={
                "convolution_layer": "Conv",
                "local_pooling_layer": "Local",
                "global_pooling_layer": "Global",
                "dataset": "Dataset",
            }
        )

    def set_datasets(self) -> None :
        """
        Set a new attribute 'datasets' that will be the different datasets
        (after the columns have been renamed)
        """
        self.datasets = self.df["Dataset"].unique()

    def pivot(self) -> None :
        """
        Use the function 'pivot' of pandas to 'group by' datasets (multicolumns)
        """
        if self.global_pooling == "":
            indexes = ["Conv", "Local", "Global"]
        else:
            indexes = ["Conv", "Local"]
        self.df = self.df.pivot(
            index=indexes,
            columns="Dataset",
            values=["accuracy", "training_time", "mean_accuracy"],
        )
    def set_ranking(self) -> None :
        """
        Set a new attribute to be the argsort of the mean accuracy (after the groupby)
        """
        self.df_ranking = self.df["mean_accuracy"].rank(ascending=False)

    def set_conv_and_pool_map(self) -> None :
        """
        Set the convolution mapping and the pooling mapping (new attributes)
        """
        self.conv_map = self.df.reset_index(level=self.level)["Conv"].to_dict()
        self.pool_map = self.df.reset_index(level=self.level)["Local"].to_dict()

    def set_aggregator(self) -> None :
        """
        Set the aggregators (two new attributes) that will be used to get
        the worst and best accuracies
        """
        if self.per_dataset:
            self.aggregator = lambda x:x
            self.aggregated_columns = self.datasets
        else:
            self.aggregator = lambda x:x.mean(axis=1)
            self.aggregated_columns = 0
    
    def set_best(self) -> None :
        """
        Set the best pooling and architecture with respect to the aggregator
        (two dataframes) plus the corresponding values
        """
        self.df_best_pooling = self.aggregator(self.df_ranking).reset_index(level=self.level) \
            .groupby(["Conv"]).idxmin()[self.aggregated_columns].copy()
        self.df_best_architecture = self.aggregator(self.df_ranking).reset_index(level=self.level) \
            .groupby(["Local"]).idxmin()[self.aggregated_columns].copy()

        self.best_indexes_by_architecture = list(set(self.df_best_pooling.values.flatten()))
        self.best_indexes_by_pooling = list(set(self.df_best_architecture.values.flatten()))

        self.df_best_pooling = self.df_best_pooling.map(lambda x:self.pool_map[x])
        self.df_best_architecture = self.df_best_architecture.map(lambda x:self.conv_map[x])

    def set_worst(self) -> None :
        """
        Set the worst pooling and architecture with respect to the aggregator
        """
        self.df_worst_pooling = \
            self.aggregator(self.df_ranking).reset_index(level=self.level).groupby(["Conv"]) \
            .idxmax()[self.aggregated_columns].copy()
        self.df_worst_architecture = \
            self.aggregator(self.df_ranking).reset_index(level=self.level).groupby(["Local"]) \
            .idxmax()[self.aggregated_columns].copy()
        
        self.df_worst_pooling = self.df_worst_pooling.map(lambda x:self.pool_map[x])
        self.df_worst_architecture = self.df_worst_architecture.map(lambda x:self.conv_map[x])

    def modify_training_time(self) -> None :
        """
        Modify the training time for self.df after all the previous changes
        """
        training_time = self.df["training_time"].sum(axis=1).copy().astype(int).astype(str)
        self.df = self.df.drop(columns=["training_time", "mean_accuracy"])
        self.df.columns = self.df.columns.droplevel(0)
        self.df = self.df.rename_axis(None, axis=1)
        self.df["Training Time"] = training_time

    def set_average_ranking(self) -> None :
        """
        Set two dataframes with the average ranking per architecture
        and pooling respectively
        """
        self.average_ranking_df_archi = self.df_ranking.reset_index(level=[0]) \
            .groupby(["Conv"])[self.datasets].min().astype(int)
        self.average_ranking_df_pooling = self.df_ranking.reset_index(level=[1]) \
            .groupby(["Local"])[self.datasets].min().astype(int)
    
    def set_df_by(self) -> Tuple[pd.DataFrame, pd.DataFrame] :
        """
        Set the dataframe with the best architectures and poolings
        respectively
        """
        self.df_by_architecture = self.df.iloc[self.best_indexes_by_architecture].copy()
        self.df_by_pooling = self.df.iloc[self.best_indexes_by_pooling].copy() \
            .reorder_levels(self.level_transpose).sort_index()
    
    def get_all(self) -> Tuple[pd.DataFrame,
                               pd.DataFrame,
                               pd.DataFrame,
                               pd.DataFrame,
                               pd.DataFrame,
                               pd.DataFrame,
                               pd.DataFrame,
                               pd.DataFrame,
                               pd.DataFrame]:
        """
        Return all the useful dataframes
        """
        return (
                self.df, 
                self.df_best_architecture,
                self.df_worst_architecture,
                self.average_ranking_df_archi,
                self.df_by_architecture,
                self.df_best_pooling,
                self.df_worst_pooling,
                self.average_ranking_df_pooling,
                self.df_by_pooling
        )
    
    def save_df_latex(df : pd.DataFrame, path_dir : str, name : str) -> None :
        """
        Save the dataframe as a tex code in a txt file given
        the name of the directory and the name of the file
        """
        os.makedirs(path_dir, exist_ok=True)
        with open(os.path.join(path_dir, name), 'w') as file:
            file.write(df.to_latex(multirow=True,index_names=False))

    def save_all(self) -> None :
        """
        Save all the dataframes in the respective save_file_names
        """
        ToTable.save_df_latex(self.df, self.save_path, self.save_file_name_df)
        ToTable.save_df_latex(self.average_ranking_df_archi, self.save_path, self.save_file_name_average_ranking_df_archi)
        ToTable.save_df_latex(self.average_ranking_df_pooling, self.save_path, self.save_file_name_average_ranking_df_pooling)
        ToTable.save_df_latex(self.df_by_architecture, self.save_path, self.save_file_name_df_by_architecture)
        ToTable.save_df_latex(self.df_by_pooling, self.save_path, self.save_file_name_df_by_pooling)
        ToTable.save_df_latex(self.df_best_architecture, self.save_path, self.save_file_name_df_best_architecture)
        ToTable.save_df_latex(self.df_worst_architecture, self.save_path, self.save_file_name_df_worst_architecture)
        ToTable.save_df_latex(self.df_best_pooling, self.save_path, self.save_file_name_df_best_pooling)
        ToTable.save_df_latex(self.df_worst_pooling, self.save_path, self.save_file_name_df_worst_pooling)