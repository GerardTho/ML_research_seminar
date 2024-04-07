from typing import List, Dict, Tuple, Any

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.markers as markers
import matplotlib.patches as mpatches

import pandas as pd

from collections import defaultdict


def get_convolution_mapping(list_dict: List[Dict]) -> Tuple[List, Dict, List]:
    """
    For the shape of the points of the following plot functions, we will
    need a mapping of the value of the architecture to the shapes

    To do so, this function return the mapping (a dictionary) and
    the list of all shapes corresponding to the architecture
    """

    # the convolutions will be used for the shape of the points
    convolutions = [d["convolution_layer"] for d in list_dict]
    unique_convolutions = np.unique(convolutions)

    if len(unique_convolutions) > 41:
        raise Exception(
            f"Not enough possible values of shape for the convolutions : got {len(convolutions)} but expected 41 at most"
        )
    # same as for the colors: creating a mapping of the convolutions to the shapes

    existing_shapes = list(markers.MarkerStyle.markers.keys())
    shape_mapping = {}

    for i, value in enumerate(unique_convolutions):
        shape_mapping[value] = existing_shapes[i]

    shapes = [shape_mapping[value] for value in convolutions]

    return convolutions, shape_mapping, shapes


def get_pooling_mapping(list_dict: List[Dict]) -> Tuple[List, Dict, List]:
    """
    For the color of the points of the following plot functions, we will
    need a mapping of the pooling to the colors

    To do so, this function return the mapping (a dictionary) and
    the list of all colors corresponding to the pooling
    """
    poolings = [d["local_pooling_layer"] for d in list_dict]
    unique_poolings = np.unique(poolings)

    existing_colors = plt.get_cmap("tab10", len(unique_poolings))

    color_mapping = {}
    # creating a dictionary that maps each value of pooling to a color

    for i, value in enumerate(unique_poolings):
        color_mapping[value] = existing_colors(i)

    # mapping the values through the dictionary
    colors = [color_mapping[value] for value in poolings]

    return poolings, color_mapping, colors


def plot_from_dict(
    list_dict: List[Dict], figsize: Tuple[int, int] = (10, 6), **kwargs
) -> None:
    """
    Plot the graph resulting from the list of dictionary

    figsize -> width and height of the figure (figsize argument matplotlib figure function)
    **kwargs -> additional keyword arguments passed to matplotlib scatter function

    Raises an error if the desired keys are not present in a dictionary
    """
    # checking that there is no problem of keys in each given dictionary
    key_values_to_check = [
        "nb_parameters",
        "mean_accuracy",
        "homophily",
        "local_pooling_layer",
        "convolution_layer",
        "dataset",
    ]

    for i, d in enumerate(list_dict):

        if not (all(key in d for key in key_values_to_check)):

            raise Exception(f"Problem of key for the {i}-th dictionary")

    # the x,y and z of the scatter in 3D
    x = np.array([d["nb_parameters"] for d in list_dict])
    y = np.array([d["mean_accuracy"] for d in list_dict])
    z = np.array([d["homophily"] for d in list_dict])

    _, color_mapping, colors = get_pooling_mapping(list_dict)

    _, shape_mapping, shapes = get_convolution_mapping(list_dict)

    # scatter in 3D
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = fig.add_subplot(111, projection="3d")

    # the legend for the colors
    colors_keys = list(color_mapping.keys())
    color_values = list(color_mapping.values())
    legend_color = [
        mpatches.Patch(color=color_values[i], label=f"{colors_keys[i]}-pool")
        for i in range(len(colors_keys))
    ]

    # the legend for the shapes
    # To do so, a trick: plotting invisible lines for each shape
    shape_keys = list(shape_mapping.keys())
    shape_values = list(shape_mapping.values())
    legend_shape = [
        ax.scatter([], [], color="black", marker=shape_values[i], label=shape_keys[i])
        for i in range(len(shape_keys))
    ]

    for x_value, y_value, z_value, color, shape in zip(x, y, z, colors, shapes):
        ax.scatter(
            x_value,
            y_value,
            z_value,
            c=np.array(color).reshape((1, 4)),
            marker=shape,
            **kwargs,
        )

    ax.legend(handles=legend_color + legend_shape, bbox_to_anchor=(1.5, 1))

    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Mean accuracy")
    ax.set_zlabel("Homophily", labelpad=0.0)

    plt.savefig("./Visualisation/results/scatter_plot_3D/scatter_plot_3D.png")

    plt.show()

    return


def get_grouped_list(list_dict: List[Dict]) -> List[List[Dict]]:
    """
    Returns a list of list of the dictionaries of list_dict, groupby
    the pooling (this will be used for the following function
    pairplot_from_dict)
    """

    grouped_data = defaultdict(list)

    # Group dictionaries by the 'pooling' key
    for d in list_dict:
        pooling_key = d.get("pooling_and_archi", None)
        grouped_data[pooling_key].append(d)

    # Convert default dict to list of lists
    grouped_list = [v for _, v in grouped_data.items()]

    return grouped_list


def pairplot_from_dict(
    list_dict: List[Dict],
    rows_to_plot: List[Tuple[str, str]],
    dim_grid_subplots: Tuple[int, int],
    figsize: Tuple[int, int] | None = None,
    plot: bool = True,
    kwargs1: Dict[str, Any] = {},
    kwargs2: Dict[str, Any] = {},
    kwargs3: Dict[str, Any] = {},
) -> None:
    """
    Plot all the variables described in rows_to_plot in subfigures


    rows_to_plot -> All the variables of the dictionnaries of list_dict we want to
    plot (for instance, denoting (x,y) the first entry of rows_to_plot, the first plot will
    be dict[x] for the x-axis and dict[y] for the y-axis, for dict in list_dict)

    dim_grid_subplots -> the dimension of the grid of subplots (for instance (2,3) means that there
    are two rows and 3 columns)

    figsize -> width and height of the figure (figsize argument matplotlib figure function)

    plot -> what kind of plot : if True, a plot, else a scatter

    **kwargs1 -> additional keyword arguments passed to matplotlib scatter function
    **kwargs2 -> additional keyword arguments passed to matplotlib legend function
    **kwargs2 -> additional keyword arguments passed to matplotlib plot function (if plot)

    Raises error if some elements of the rows don't correspond to key values in list_dict,
    or if the number of elements of rows_to_plot don't fit the dimension of dim_grid_subplots
    """
    # the list of all keys present in list_dict
    keys_list_dict = [dic.keys() for dic in list_dict]

    for i, (key1, key2) in enumerate(rows_to_plot):

        # check if key1 is present in the keys of all dictionary
        if not (all(key1 in key for key in keys_list_dict)):

            raise Exception(
                f"The first key of the {i+1}-th element of the variables "
                "to plot is not present in all keys of the dictionnaries "
            )

        # same for key2
        if not (all(key2 in key for key in keys_list_dict)):

            raise Exception(
                f"The first key of the {i+1}-th element of the variables "
                "to plot is not present in all keys of the dictionnaries "
            )

    n_plot_rows, n_plot_cols = dim_grid_subplots

    if n_plot_rows * n_plot_cols != len(rows_to_plot):

        raise Exception(
            f"The number of plots imposed by the dimension of the grid "
            f"({n_plot_rows} x {n_plot_cols} = {n_plot_rows*n_plot_cols}) "
            f"is not consistent with the number of plots ({len(rows_to_plot)}) "
        )

    _, color_mapping, colors = get_pooling_mapping(list_dict)
    _, shape_mapping, shapes = get_convolution_mapping(list_dict)

    colors_keys = list(color_mapping.keys())
    color_values = list(color_mapping.values())

    shape_keys = list(shape_mapping.keys())
    shape_values = list(shape_mapping.values())

    _, axs = plt.subplots(n_plot_rows, n_plot_cols, figsize=figsize)

    axs = axs.flatten()

    if plot:
        grouped_list = get_grouped_list(list_dict)
        n_grouped_list = len(grouped_list)

    for ax, (key1, key2) in zip(axs, rows_to_plot):

        legend_color = [
            mpatches.Patch(color=color_values[i], label=f"{colors_keys[i]}-pool")
            for i in range(len(colors_keys))
        ]
        legend_shape = [
            ax.scatter(
                [], [], color="black", marker=shape_values[i], label=shape_keys[i]
            )
            for i in range(len(shape_keys))
        ]

        x_values = [dic[key1] for dic in list_dict]
        y_values = [dic[key2] for dic in list_dict]

        for x_value, y_value, color, shape in zip(x_values, y_values, colors, shapes):
            ax.scatter(
                x_value,
                y_value,
                c=np.array(color).reshape((1, 4)),
                marker=shape,
                **kwargs1,
            )

        if plot:
            for i in range(n_grouped_list):

                x_plot = np.array([dic[key1] for dic in grouped_list[i]])
                y_plot = np.array([dic[key2] for dic in grouped_list[i]])

                plot_pooling = grouped_list[i][0]["local_pooling_layer"]
                color = color_mapping[plot_pooling]

                x_plot_argsort = np.argsort(x_plot)
                x_plot = x_plot[x_plot_argsort]
                y_plot = y_plot[x_plot_argsort]

                ax.plot(
                    x_plot,
                    y_plot,
                    c=np.array(color).reshape((1, 4)),
                    **kwargs3,
                )

        ax.legend(handles=legend_color + legend_shape, **kwargs2)

        ax.set_xlabel(key1)
        ax.set_ylabel(key2)

    plt.tight_layout()

    plt.savefig("./Visualisation/results/pairplot/pairplot.png")

    # Show the plot
    plt.show()

    return

def to_table(list_dict: List[Dict], per_dataset=False) -> pd.DataFrame:
    """
    Convert a list of dictionaries into a formatted Pandas DataFrame for tabular presentation.

    Parameters:
        list_dict (List[Dict]): A list of dictionaries containing the following keys:
            - 'dataset': The name of the dataset.
            - 'global_pooling_layer': The type of global pooling layer used.
            - 'local_pooling_layer': The type of local pooling layer used.
            - 'mean_accuracy': The mean accuracy value.
            - 'std_accuracy': The standard deviation of accuracy.

    Returns:
        pd.DataFrame: A formatted Pandas DataFrame with columns 'dataset', 'pooling_layer', and 'accuracy'.
            - 'dataset': Name of the dataset.
            - 'pooling_layer': Concatenation of 'local_pooling_layer' and 'global_pooling_layer'.
            - 'accuracy': Formatted string of mean accuracy ± standard deviation.

    Example Usage:
        list_dict = [
            {"dataset": "Dataset A", "global_pooling_layer": "Avg", "local_pooling_layer": "Max", "mean_accuracy": 0.85, "std_accuracy": 0.03},
            {"dataset": "Dataset B", "global_pooling_layer": "Sum", "local_pooling_layer": "Avg", "mean_accuracy": 0.92, "std_accuracy": 0.02}
        ]
        df = to_table(list_dict)
        print(df)

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

    for dic in list_dict:
        dic_results["dataset"].append(dic["dataset"])
        dic_results["global_pooling_layer"].append(dic["global_pooling_layer"])
        dic_results["local_pooling_layer"].append(dic["local_pooling_layer"])
        dic_results["convolution_layer"].append(dic["convolution_layer"])
        dic_results["mean_accuracy"].append(dic["mean_accuracy"])
        dic_results["std_accuracy"].append(dic["std_accuracy"])
        training_time = 0
        for i in range(1, 11):
            try:
                training_time += (
                    dic["split " + str(i)]["train_time_per_epoch"]
                    * dic["split " + str(i)]["last_epoch"]
                )
            except KeyError:
                training_time += 0
        dic_results["training_time"].append(training_time / 10)

    df = pd.DataFrame(dic_results)

    df = df[df["dataset"]!="NCI1"]

    df["local_pooling_layer"] = df["local_pooling_layer"].astype(str)
    indexes_to_bold = df.groupby("dataset")["mean_accuracy"].idxmax()
    df["accuracy"] = (
        "$"
        + df["mean_accuracy"].apply("{:.3f}".format).astype(str)
        + "\pm"
        + df["std_accuracy"].apply("{:.3f}".format).astype(str)
        + "$"
    )
    df.loc[indexes_to_bold, "accuracy"] = (
        "$\\bm{"
        + df.loc[indexes_to_bold, "mean_accuracy"].apply("{:.3f}".format).astype(str)
        + "\pm"
        + df.loc[indexes_to_bold, "std_accuracy"].apply("{:.3f}".format).astype(str)
        + "}$"
    )
    df = df.rename(
        columns={
            "convolution_layer": "Conv",
            "local_pooling_layer": "Local",
            "global_pooling_layer": "Global",
            "dataset": "Dataset",
        }
    )
    datasets = df["Dataset"].unique()
    df = df.pivot(
        index=["Conv", "Local", "Global"],
        columns="Dataset",
        values=["accuracy", "training_time", "mean_accuracy"],
    )


    df_ranking = df["mean_accuracy"].rank(ascending=False)#.groupby(["Conv", "Local"], group_keys=True)[datasets].mean()
    conv_map = df.reset_index(level=[0,1,2])["Conv"].to_dict()
    pool_map = df.reset_index(level=[0,1,2])["Local"].to_dict()
    average_ranking_df_archi = df_ranking.reset_index(level=[0]).groupby(["Conv"])[datasets].mean().astype(int)
    average_ranking_df_pooling = df_ranking.reset_index(level=[1]).groupby(["Local"])[datasets].mean().astype(int)

    if per_dataset:
        aggregator = lambda x:x
        aggregated_columns = datasets
    else:
        aggregator = lambda x:x.mean(axis=1)
        aggregated_columns = 0
        
    df_best_pooling = aggregator(df_ranking).reset_index(level=[0,1,2]).groupby(["Conv"]).idxmin()[aggregated_columns].copy()
    df_best_architecture = aggregator(df_ranking).reset_index(level=[0,1,2]).groupby(["Local"]).idxmin()[aggregated_columns].copy()
    best_indexes_by_architecture = list(set(df_best_pooling.values.flatten()))
    best_indexes_by_pooling = list(set(df_best_architecture.values.flatten()))
    df_best_pooling =  df_best_pooling.map(lambda x:pool_map[x])
    df_best_architecture =  df_best_architecture.map(lambda x:conv_map[x])

    df_worst_pooling = aggregator(df_ranking).reset_index(level=[0,1,2]).groupby(["Conv"]).idxmax()[aggregated_columns].copy()
    df_worst_architecture = aggregator(df_ranking).reset_index(level=[0,1,2]).groupby(["Local"]).idxmax()[aggregated_columns].copy()
    df_worst_pooling =  df_worst_pooling.map(lambda x:pool_map[x])
    df_worst_architecture =  df_worst_architecture.map(lambda x:conv_map[x])

    training_time = df["training_time"].sum(axis=1).copy().astype(int).astype(str)
    df = df.drop(columns=["training_time", "mean_accuracy"])
    df.columns = df.columns.droplevel(0)
    df = df.rename_axis(None, axis=1)
    df["Training Time"] = training_time

    df_by_architecture = df.iloc[best_indexes_by_architecture].copy()
    df_by_pooling = df.iloc[best_indexes_by_pooling].copy()
    
    return df, df_best_architecture, df_worst_architecture, average_ranking_df_archi, df_by_architecture, df_best_pooling, df_worst_pooling, average_ranking_df_pooling, df_by_pooling.reorder_levels([1,2,0]).sort_index()

def get_mean_tuple_list(tuple_list: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """
    For a list of tuple, returns a dictionary with the unique values of str
    as keys and the mean of values as values
    """
    res = defaultdict(lambda: (0, 0))
    for val, key in tuple_list:
        prev_val, i = res[key]
        res[key] = (prev_val + val, i + 1)

    get_mean = lambda t: t[0] / t[1]

    return {key: get_mean(value) for key, value in res.items()}


def plot_bar_dataset(
    list_dict: List[Dict],
    groupby: str,
    stack: str | None,
    x_figsize: int,
    bar_width: float,
    offset: float = 0.0,
    cmap: str = "tab10",
    n_colors: int = 10,
    kwargs1: Dict[str, Any] = {},
    kwargs2: Dict[str, Any] = {},
):
    """
    groupby -> the key along which each bar will be plotted
    stack -> the key along which each bar will be duplicated
        (if None, there is no stacking)
    """

    # first, create a dictionary whose keys are the dataset and values are
    # the list of all element from list_dict for this dataset
    data_by_dataset = {}

    for entry in list_dict:

        dataset = entry["dataset"]

        if dataset not in data_by_dataset:
            data_by_dataset[dataset] = []

        data_by_dataset[dataset].append(entry)

    # the colors of the bars
    colors = [plt.get_cmap(cmap)(i) for i in range(n_colors)]

    # a subplot for each dataset
    _, axes = plt.subplots(
        nrows=len(data_by_dataset),
        ncols=1,
        figsize=(x_figsize, len(data_by_dataset) * 5),
        squeeze=False,
    )
    axes = axes.flatten()
    # this doesn't change much, the value is arbitrary

    # loop for each plot (dataset)
    for ax, (dataset, records) in zip(axes, data_by_dataset.items()):

        groupby_values = np.unique([record[groupby] for record in records])
        # the keys of this dictionary are the groupby values (value for each group
        # of barplot)
        mean_accuracies = {e: [] for e in groupby_values}

        for record in records:
            e = record[groupby]
            if stack is None:
                mean_accuracies[e].append(record["mean_accuracy"])
            else:
                # For the moment, mean_accuracies is a dictionary where each value
                # is the tuple of the mean accuracy and the str corresponding to the
                # stack variable
                mean_accuracies[e].append((record["mean_accuracy"], record[stack]))

        if stack is None:
            # the list of the mean accuracy for each groupby variable
            mean_accuracies = list(
                map(lambda l: sum(l) / len(l), mean_accuracies.values())
            )
        else:
            # a dictionary of dictionary where for each dictionary, the key is the stack variable
            # and the value is the average accuracy along each identical element across both the
            # stack and groupby variable
            mean_accuracies = {
                key: get_mean_tuple_list(value)
                for key, value in mean_accuracies.items()
            }

        # repeating colors if necessary
        if len(groupby_values) > n_colors:
            colors = colors * (len(groupby_values) // n_colors + 1)

        # if no stack variables
        if stack is None:
            # bars for each groupby values
            bars = ax.bar(
                groupby_values,
                mean_accuracies,
                color=colors[: len(groupby_values)],
                **kwargs1,
            )

            ax.set_xticks(range(len(groupby_values)))

            # annotating the accuracies on the center of each bar
            for bar, acc in zip(bars, mean_accuracies):
                height = bar.get_height()
                ax.annotate(
                    f"{acc:.3f}",
                    (bar.get_x() + bar.get_width() / 2, height / 2),
                    textcoords="offset points",
                    xytext=(0, 0),
                    ha="center",
                    va="center",
                    rotation=0,
                )

        # if there is a stack variable
        else:
            # keys : each groupby value
            keys = list(mean_accuracies.keys())
            # values : the keys of the dictionaries (among the dictionary)
            values = list(mean_accuracies[keys[0]].keys())
            # useful for xticks afterwards
            index = np.arange(len(keys))

            # loop across the keys of dictionaries
            for i, value in enumerate(values):
                # the mean accuracies for the tuple (groupby, stack)
                accuracies = [mean_accuracies[key][value] for key in keys]
                ax.bar(
                    index + i * bar_width,
                    accuracies,
                    bar_width,
                    color=colors[i],
                    label=value,
                    **kwargs1,
                )

                # print the mean accuracy in the middle of the bars
                offset_val = offset if len(keys) > 1 else 0.0
                for j, accuracy in enumerate(accuracies):
                    ax.annotate(
                        f"{accuracy:.3f}",
                        (index[j] + i * bar_width - offset_val, accuracy / 2),
                        textcoords="offset points",
                        xytext=(0, 0),
                        ha="center",
                        va="center",
                        rotation=270,
                    )

            ax.set_xticks(index + (len(values) - 1) * bar_width / 2)
            ax.set_xticklabels(keys)
            ax.legend(**kwargs2)

        ax.set_title(f"Dataset: {dataset}")
        ax.set_ylabel("Mean Accuracy")
        ax.set_xlabel(groupby[0].upper() + groupby[1:])

        # Save the current subplot
        # We need to draw the canvas to ensure that all elements are laid out correctly
        plt.gcf().canvas.draw()

        # Get the bounding box of the axis, including any labels, titles, etc.
        bbox = ax.get_tightbbox(plt.gcf().canvas.get_renderer())
        bbox_inches = bbox.transformed(plt.gcf().dpi_scale_trans.inverted())

        # Save the subplot using the bounding box
        plt.savefig(f"./Visualisation/results/barplot/barplot-groupby_{groupby}"
                    f"-stack_{stack}-dataset_{dataset}.png", bbox_inches=bbox_inches)

    # Adjust the spacing after creating subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(
            f"./Visualisation/results/barplot/barplot-groupby_{groupby}"
            f"-stack_{stack}-aggregate.png"
            )

    plt.show()


def plot_losses(list_dict, train="train"):
    for dict in list_dict:
        if "split 1" in dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(
                np.arange(len(dict["split 1"]["train_losses"])),
                dict["split 1"][train + "_losses"],
            )
            ax.set_xlabel("Epochs")
            ax.set_ylabel(train + " loss")
            if dict["local_pooling_layer"] is not None:
                ax.set_title(
                    train
                    + " loss across epochs with early stopping"
                    + "\n"
                    + "for "
                    + dict["dataset"]
                    + " with "
                    + dict["convolution_layer"]
                    + ","
                    + dict["local_pooling_layer"]
                    + " and "
                    + dict["global_pooling_layer"]
                )
                plt.savefig(
                    "./Visualisation/results/losses/"
                    + train
                    + "/"
                    + dict["dataset"]
                    + "_"
                    + dict["convolution_layer"]
                    + "_"
                    + dict["local_pooling_layer"]
                    + "_"
                    + dict["global_pooling_layer"]
                    + ".png"
                )
                plt.close()
            else:
                ax.set_title(
                    train
                    + " loss across epochs with early stopping"
                    + "\n"
                    + "for "
                    + dict["dataset"]
                    + " with "
                    + dict["convolution_layer"]
                    + ","
                    + "None "
                    + " and "
                    + dict["global_pooling_layer"]
                )
                plt.savefig(
                    "./Visualisation/results/losses/"
                    + train
                    + "/"
                    + dict["dataset"]
                    + "_"
                    + dict["convolution_layer"]
                    + "_"
                    + "None_"
                    + dict["global_pooling_layer"]
                    + ".png"
                )
                plt.close()


def sort_list_dict(list_dict):
    """Sort the list of dictionaries in alphabetical order with respect to
    architecture_localpooling_globalpooling

    Args:
        list_dict (list): list of dictionaries of runs

    Returns:
        list: sorted list
    """
    list_dict__architecture_as_key = {
        f"{d['convolution_layer']}_{d['local_pooling_layer']}_{d['global_pooling_layer']}_{d['dataset']}": d
        for d in list_dict
    }
    list_dict__architecture_as_key = dict(
        sorted(list_dict__architecture_as_key.items())
    )
    return list(list_dict__architecture_as_key.values())


def get_worst_best_acc_per_dataset(list_dict, datasets):
    best_accs = {}
    worst_accs = {}
    for d in list_dict:
        dataset = d.get("dataset")
        mean = d.get("mean_accuracy")
        if dataset not in best_accs.keys():
            best_accs[dataset] = (mean, d)
            worst_accs[dataset] = (mean, d)
        else:
            best_mean_dataset = best_accs[dataset][0]
            if mean > best_mean_dataset:
                best_accs[dataset] = (mean, d)
            worst_mean_dataset = worst_accs[dataset][0]
            if mean < worst_mean_dataset:
                worst_accs[dataset] = (mean, d)
    return worst_accs, best_accs


def plot_acc_parameters(list_dict):
    if not list_dict:
        print("Error: Empty list")
        return

    list_dict = sort_list_dict(list_dict)
    datasets = set(d["dataset"] for d in list_dict)
    worst_acc_per_dataset, best_acc_per_dataset = get_worst_best_acc_per_dataset(
        list_dict, datasets
    )

    for dataset in datasets:
        plt.figure(figsize=(8, 6))
        colors = plt.cm.hsv(np.linspace(0, 1, 100))  # Define 15 different colors
        color_map = plt.cm.colors.ListedColormap(colors)  # Create a custom colormap
        nb_parameters = []
        mean_accuracy = []
        labels = set()
        for split in list_dict:
            if split.get("dataset") == dataset:
                nb_parameters.append(split.get("nb_parameters", 0))
                mean_accuracy.append(split.get("mean_accuracy", 0))
                label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                labels.add(label)

        sorted_labels = sorted(labels)  # Sort labels alphabetically

        colors = [color_map(i / len(sorted_labels)) for i in range(len(sorted_labels))]
        label_color_dict = {label: color for label, color in zip(sorted_labels, colors)}

        for split in list_dict:
            if split.get("dataset") == dataset:
                label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                color = label_color_dict.get(
                    label, "black"
                )  # Use black color for labels not in sorted_labels
                plt.scatter(
                    split.get("nb_parameters", 0),
                    split.get("mean_accuracy", 0),
                    label=label if label in sorted_labels else None,
                    color=color,
                    alpha=0.6,
                    marker="^" if split.get("global_pooling_layer") == "mean" else "o",
                )
                # check if this corresponds to the best/worst architecture
                if split == best_acc_per_dataset[dataset][1]:
                    plt.text(
                        split.get("nb_parameters", 0) + 0.001,
                        split.get("mean_accuracy", 0) + 0.001,
                        f"{label}: {split.get('mean_accuracy', 0):.3f}",
                        fontsize=9,
                    )
                if split == worst_acc_per_dataset[dataset][1]:
                    plt.text(
                        split.get("nb_parameters", 0) - 0.001,
                        split.get("mean_accuracy", 0) - 0.001,
                        f"{label}: {split.get('mean_accuracy', 0):.3f}",
                        fontsize=9,
                    )

        plt.xlabel("Number of Parameters")
        plt.ylabel("Mean Accuracy")
        plt.title(
            f"Mean Accuracy vs Number of Parameters for {dataset} \n Triangle = mean ending pooling | Circle = max ending pooling"
        )
        plt.legend(
            loc="lower right",
            bbox_to_anchor=(1.4, 0),
            borderaxespad=0.0,
            prop={"size": 10},
        )
        plt.savefig(
            f"./Visualisation/results/acc_parameters/{dataset}.png", bbox_inches="tight"
        )
        plt.close()


def plot_acc_time_epoch(
    list_dict,
    time_per_epoch=True,
    xlabel="Train time per epoch (s)",
    title="Mean Accuracy vs Train time per epoch",
    save_dir="acc_train_time_per_epoch",
):
    if not list_dict:
        print("Error: Empty list")
        return

    list_dict = sort_list_dict(list_dict)
    datasets = set(d["dataset"] for d in list_dict)
    worst_acc_per_dataset, best_acc_per_dataset = get_worst_best_acc_per_dataset(
        list_dict, datasets
    )
    for dataset in datasets:
        plt.figure(figsize=(8, 6))
        colors = plt.cm.hsv(np.linspace(0, 1, 100))  # Define 15 different colors
        color_map = plt.cm.colors.ListedColormap(colors)  # Create a custom colormap
        train_time_per_epoch = []
        mean_accuracy = []
        labels = set()
        for split in list_dict:
            if "split 1" in split:
                if split.get("dataset") == dataset:
                    train_time_per_epoch.append(
                        split["split 1"].get("train_time_per_epoch", 0)
                    )
                    mean_accuracy.append(split.get("mean_accuracy", 0))
                    label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                    labels.add(label)

        sorted_labels = sorted(labels)  # Sort labels alphabetically

        colors = [color_map(i / len(sorted_labels)) for i in range(len(sorted_labels))]
        label_color_dict = {label: color for label, color in zip(sorted_labels, colors)}

        for split in list_dict:
            if "split 1" in split:
                if split.get("dataset") == dataset:
                    label = f"{split.get('convolution_layer')}_{split.get('local_pooling_layer', 'None')}"
                    color = label_color_dict.get(
                        label, "black"
                    )  # Use black color for labels not in sorted_labels
                    time_ = (
                        split["split 1"].get("train_time_per_epoch", 0)
                        if time_per_epoch
                        else split["split 1"].get("train_time_per_epoch", 0)
                        * len(split["split 1"]["train_losses"])
                        / 60
                    )
                    plt.scatter(
                        time_,
                        split.get("mean_accuracy", 0),
                        label=label if label in sorted_labels else None,
                        color=color,
                        alpha=0.6,
                        marker=(
                            "^" if split.get("global_pooling_layer") == "mean" else "o"
                        ),
                    )
                    # check if this corresponds to the best architecture
                    if split == best_acc_per_dataset[dataset][1]:
                        plt.text(
                            time_ + 0.001,
                            split.get("mean_accuracy", 0) + 0.001,
                            f"{label}: {split.get('mean_accuracy', 0):.3f}",
                            fontsize=9,
                        )
                    if split == worst_acc_per_dataset[dataset][1]:
                        plt.text(
                            time_ - 0.001,
                            split.get("mean_accuracy", 0) - 0.001,
                            f"{label}: {split.get('mean_accuracy', 0):.3f}",
                            fontsize=9,
                        )

        plt.xlabel(xlabel)
        plt.ylabel("Mean Accuracy")
        plt.title(
            f"{title} for {dataset} \n Triangle = mean ending pooling | Circle = max ending pooling"
        )
        plt.legend(
            loc="lower right",
            bbox_to_anchor=(1.4, 0),
            borderaxespad=0.0,
            prop={"size": 10},
        )
        plt.savefig(
            f"./Visualisation/results/{save_dir}/{dataset}.png", bbox_inches="tight"
        )
        plt.close()


def plot_acc_full_train_time(list_dict):
    plot_acc_time_epoch(
        list_dict,
        time_per_epoch=False,
        xlabel="Full train time (min)",
        title="Mean Accuracy vs Full train time",
        save_dir="acc_full_train_time",
    )


def plot_acc(list_dict, train="train"):
    for dict in list_dict:
        if "split 1" in dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(
                np.arange(len(list_dict[0]["split 1"]["train_accuracies"])),
                list_dict[0]["split 1"][train + "_accuracies"],
            )
            ax.set_xlabel("Epochs")
            ax.set_ylabel(train + " acc")
            if dict["local_pooling_layer"] is not None:
                ax.set_title(
                    train
                    + " accuracy across epochs with early stopping"
                    + "\n"
                    + "for "
                    + dict["dataset"]
                    + " with "
                    + dict["convolution_layer"]
                    + ","
                    + dict["local_pooling_layer"]
                    + " and "
                    + dict["global_pooling_layer"]
                )
                plt.savefig(
                    "./Visualisation/results/acc/"
                    + train
                    + "/"
                    + dict["dataset"]
                    + "_"
                    + dict["convolution_layer"]
                    + "_"
                    + dict["local_pooling_layer"]
                    + "_"
                    + dict["global_pooling_layer"]
                    + ".png"
                )
                plt.close()
            else:
                ax.set_title(
                    train
                    + " accuracy across epochs with early stopping"
                    + "\n"
                    + "for "
                    + dict["dataset"]
                    + " with "
                    + dict["convolution_layer"]
                    + ","
                    + "None "
                    + " and "
                    + dict["global_pooling_layer"]
                )
                plt.savefig(
                    "./Visualisation/results/acc/"
                    + train
                    + "/"
                    + dict["dataset"]
                    + "_"
                    + dict["convolution_layer"]
                    + "_"
                    + "None_"
                    + dict["global_pooling_layer"]
                    + ".png"
                )
                plt.close()


def plot_acc_and_loss(list_dict, train="train"):
    for dict in list_dict:
        if "split 1" in dict:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            color = "tab:blue"
            ax1.plot(
                np.arange(len(dict["split 1"]["train_accuracies"])),
                dict["split 1"][train + "_accuracies"],
                color=color,
                label=train + " accuracy",
            )
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel(train + " acc", color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            # Creating a secondary y-axis for the second curve
            ax2 = ax1.twinx()
            color = "tab:red"
            ax2.plot(
                np.arange(len(dict["split 1"][train + "_losses"])),
                dict["split 1"][train + "_losses"],
                color=color,
                label=train + " loss",
            )
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel(train + " loss", color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            # Adding legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")

            if dict["local_pooling_layer"] is not None:
                ax1.set_title(
                    train
                    + " accuracy and loss across epochs with early stopping"
                    + "\n"
                    + "for "
                    + dict["dataset"]
                    + " with "
                    + dict["convolution_layer"]
                    + ","
                    + dict["local_pooling_layer"]
                    + " and "
                    + dict["global_pooling_layer"]
                )
                plt.tight_layout()
                plt.savefig(
                    "./Visualisation/results/acc_and_loss/"
                    + train
                    + "/"
                    + dict["dataset"]
                    + "_"
                    + dict["convolution_layer"]
                    + "_"
                    + dict["local_pooling_layer"]
                    + "_"
                    + dict["global_pooling_layer"]
                    + ".png"
                )
                plt.close()
            else:
                ax1.set_title(
                    train
                    + " accuracy and loss across epochs with early stopping"
                    + "\n"
                    + "for "
                    + dict["dataset"]
                    + " with "
                    + dict["convolution_layer"]
                    + ","
                    + "None "
                    + " and "
                    + dict["global_pooling_layer"]
                )
                plt.tight_layout()
                plt.savefig(
                    "./Visualisation/results/acc_and_loss/"
                    + train
                    + "/"
                    + dict["dataset"]
                    + "_"
                    + dict["convolution_layer"]
                    + "_"
                    + "None_"
                    + dict["global_pooling_layer"]
                    + ".png"
                )
                plt.close()
