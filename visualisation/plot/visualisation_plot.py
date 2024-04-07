from typing import List, Dict, Tuple, Any

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.markers as markers
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox

from collections import defaultdict
import os


class VisualisationPlot() :

    def __init__(self, list_dict : List[Dict]) :
        
        self.path_scatter_plot = os.path.join('results', 'scatter_plot_3D', 'scatter_plot_3D.png')
        self.path_pairplot = os.path.join('results', 'pairplot')
        self.path_barplot = os.path.join('results', 'barplot')

        self.list_dict = list_dict

        # a dictionary that renames some keys for better visual results
        self.rename_dict = {
                'mean_accuracy' : 'Mean accuracy',
                'homophily' : 'Homophily',
                'nb_parameters' : 'Number of parameters',
                'avg_nodes' : 'Average number of nodes',
                'avg_edges' : 'Average number of edges',
                'local_pooling_layer' : 'Local pooling layer',
                'convolution_layer' : 'Convolution layer',
            }
                
        self.set_convolution_mapping()
        self.set_pooling_mapping()
        self.set_grouped_list()


    def set_convolution_mapping(self) -> None :
        """
        For the shape of the points of the following plot functions, we will
        need a mapping of the value of the architecture to the shapes

        To do so, this function set the mapping (a dictionary) and
        the list of all shapes corresponding to the architecture
        """

        # the convolutions will be used for the shape of the points
        convolutions = [d["convolution_layer"] for d in self.list_dict]
        unique_convolutions = np.unique(convolutions)

        if len(unique_convolutions) > 41:
            raise Exception(
                f"Not enough possible values of shape for the convolutions : got {len(convolutions)} but expected 41 at most"
            )
        
        # same as for the colors: creating a mapping of the convolutions to the shapes
        existing_shapes = ['o', 's', '^', '>', '<', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
        shape_mapping = {}

        for i, value in enumerate(unique_convolutions):
            shape_mapping[value] = existing_shapes[i]

        shapes = [shape_mapping[value] for value in convolutions]

        self.convolutions = convolutions
        self.shape_mapping = shape_mapping
        self.shapes = shapes


    def set_pooling_mapping(self) -> None :
        """
        For the color of the points of the following plot functions, we will
        need a mapping of the pooling to the colors

        To do so, this function set the mapping (a dictionary) and
        the list of all colors corresponding to the pooling
        """
        poolings = [d["local_pooling_layer"] for d in self.list_dict]
        unique_poolings = np.unique(poolings)

        colors = plt.cm.hsv(np.linspace(0, 1, 100))
        color_map = plt.cm.colors.ListedColormap(colors)
        existing_colors = [color_map(i / len(unique_poolings)) for i in range(len(unique_poolings))]

        color_mapping = {}
        # creating a dictionary that maps each value of pooling to a color

        for i, value in enumerate(unique_poolings):
            color_mapping[value] = existing_colors[i]

        # mapping the values through the dictionary
        colors = [color_mapping[value] for value in poolings]

        self.poolings = poolings
        self.color_mapping = color_mapping
        self.colors = colors


    def set_grouped_list(self) -> None :
        """
        Set a list of list of the dictionaries of list_dict, groupby
        the pooling (this will be used for the following function
        pairplot_from_dict)
        """
        grouped_data = defaultdict(list)

        # Group dictionaries by the 'pooling' key
        for d in self.list_dict:
            pooling_key = d.get("pooling_and_archi", None)
            grouped_data[pooling_key].append(d)

        # Convert default dict to list of lists
        grouped_list = [v for _, v in grouped_data.items()]

        self.grouped_list = grouped_list


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
    

    def get_list_dict_dataset(self, dataset : str | None) -> List[Dict]:
        """
        Returns the list dict but only for a given dataset (or just list_dict
        if dataset is None)
        """
        if dataset is None :
            return self.list_dict
        
        possible_dataset = set([dic['dataset'] for dic in self.list_dict])
        if not dataset in possible_dataset :
            raise Exception(
                f"The dataset {dataset} provided is not in the existing dataset {possible_dataset}"
            )
        
        return [dic for dic in self.list_dict if dic['dataset'] == dataset]
    

    def plot_from_dict(
        self, figsize: Tuple[int, int] = (10, 6), **kwargs
    ) -> None:
        """
        Plot the graph resulting from the list of dictionary

        figsize -> width and height of the figure (figsize argument matplotlib figure function)
        **kwargs -> additional keyword arguments passed to matplotlib scatter function
        """
        # the x,y and z of the scatter in 3D
        x = np.array([d["nb_parameters"] for d in self.list_dict])
        y = np.array([d["mean_accuracy"] for d in self.list_dict])
        z = np.array([d["homophily"] for d in self.list_dict])

        # scatter in 3D
        fig = plt.figure(figsize=figsize)
        plt.tight_layout()
        ax = fig.add_subplot(111, projection="3d")

        # the legend for the colors
        colors_keys = list(self.color_mapping.keys())
        color_values = list(self.color_mapping.values())
        legend_color = [
            mpatches.Patch(color=color_values[i], label=f"{colors_keys[i]}-pool")
            for i in range(len(colors_keys))
        ]

        # the legend for the shapes
        # To do so, a trick: plotting invisible lines for each shape
        shape_keys = list(self.shape_mapping.keys())
        shape_values = list(self.shape_mapping.values())
        legend_shape = [
            ax.scatter([], [], color="black", marker=shape_values[i], label=shape_keys[i])
            for i in range(len(shape_keys))
        ]

        for x_value, y_value, z_value, color, shape in zip(x, y, z, self.colors, self.shapes):
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

        plt.savefig(self.path_scatter_plot)

        plt.show()

        return


    def pairplot_from_dict(
        self,
        rows_to_plot: List[Tuple[str, str]],
        dataset : str | None,
        dim_grid_subplots: Tuple[int, int],
        figsize: Tuple[int, int] | None = None,
        plot: bool = True,
        kwargs1: Dict[str, Any] = {},
        kwargs2: Dict[str, Any] = {},
        kwargs3: Dict[str, Any] = {},
        padding_subplots : float = 0.07
    ) -> None:
        """
        Plot all the variables described in rows_to_plot in subfigures


        rows_to_plot -> All the variables of the dictionnaries of list_dict we want to
        plot (for instance, denoting (x,y) the first entry of rows_to_plot, the first plot will
        be dict[x] for the x-axis and dict[y] for the y-axis, for dict in list_dict)

        dataset -> whether to consider the whole data (None) or just the data for a given dataset

        dim_grid_subplots -> the dimension of the grid of subplots (for instance (2,3) means that there
        are two rows and 3 columns)

        figsize -> width and height of the figure (figsize argument matplotlib figure function)

        plot -> what kind of plot : if True, a plot, else a scatter

        **kwargs1 -> additional keyword arguments passed to matplotlib scatter function
        **kwargs2 -> additional keyword arguments passed to matplotlib legend function
        **kwargs2 -> additional keyword arguments passed to matplotlib plot function (if plot)

        padding_subplots -> the padding to save the box for each subplot

        Raises error if some elements of the rows don't correspond to key values in list_dict,
        or if the number of elements of rows_to_plot don't fit the dimension of dim_grid_subplots
        """
        list_dict = self.get_list_dict_dataset(dataset)

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

        # doesn't change anything if we don't consider all the dataset
        colors_keys = list(self.color_mapping.keys())
        color_values = list(self.color_mapping.values())

        shape_keys = list(self.shape_mapping.keys())
        shape_values = list(self.shape_mapping.values())

        _, axs = plt.subplots(n_plot_rows, n_plot_cols, figsize=figsize)

        axs = axs.flatten()

        if plot:
            n_grouped_list = len(self.grouped_list)

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

            for x_value, y_value, color, shape in zip(x_values, y_values, self.colors, self.shapes):
                ax.scatter(
                    x_value,
                    y_value,
                    c=np.array(color).reshape((1, 4)),
                    marker=shape,
                    **kwargs1,
                )

            if plot:
                for i in range(n_grouped_list):

                    x_plot = np.array([dic[key1] for dic in self.grouped_list[i]])
                    y_plot = np.array([dic[key2] for dic in self.grouped_list[i]])

                    plot_pooling = self.grouped_list[i][0]["local_pooling_layer"]
                    color = self.color_mapping[plot_pooling]

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

            ax.set_xscale('log')

            ax.set_xlabel(self.rename_dict.get(key1,key1))
            ax.set_ylabel(self.rename_dict.get(key2,key2))

            # Save the current subplot
            # We need to draw the canvas to ensure that all elements are laid out correctly
            plt.gcf().canvas.draw()

            # Get the bounding box of the axis, including any labels, titles, etc.
            bbox = ax.get_tightbbox(plt.gcf().canvas.get_renderer())
            bbox_inches = bbox.transformed(plt.gcf().dpi_scale_trans.inverted())

            bbox_inches_expanded = Bbox.from_extents(
                bbox_inches.x0 - padding_subplots,
                bbox_inches.y0 - padding_subplots,
                bbox_inches.x1 + padding_subplots,
                bbox_inches.y1 + padding_subplots
            )

            # Save the subplot using the bounding box
            plt.savefig(os.path.join(self.path_pairplot,
                                     f"pairplot-{key1}-{key2}.png"),
                        bbox_inches=bbox_inches_expanded)

        plt.tight_layout()
        plt.savefig(os.path.join(self.path_pairplot,
                                 f'pairplot_{dataset}.png')
        )
        plt.show()

        return



    def plot_bar_dataset(
        self,
        groupby: str,
        stack: str | None,
        x_figsize: int,
        bar_width: float,
        offset: float = 0.0,
        cmap: str = "tab10",
        n_colors: int = 10,
        kwargs1: Dict[str, Any] = {},
        kwargs2: Dict[str, Any] = {},
        padding_subplots : float = .005
    ) -> None :
        """
        groupby -> the key along which each bar will be plotted
        stack -> the key along which each bar will be duplicated
            (if None, there is no stacking)

        padding_subplots -> the padding to save the box for each subplot
        """

        # first, create a dictionary whose keys are the dataset and values are
        # the list of all element from list_dict for this dataset
        data_by_dataset = {}

        for entry in self.list_dict:

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
                    key: VisualisationPlot.get_mean_tuple_list(value)
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
            ax.set_xlabel(self.rename_dict.get(groupby,groupby))

            # Save the current subplot
            # We need to draw the canvas to ensure that all elements are laid out correctly
            plt.gcf().canvas.draw()

            # Get the bounding box of the axis, including any labels, titles, etc.
            bbox = ax.get_tightbbox(plt.gcf().canvas.get_renderer())
            bbox_inches = bbox.transformed(plt.gcf().dpi_scale_trans.inverted())

            bbox_inches_expanded = Bbox.from_extents(
                bbox_inches.x0 - padding_subplots,
                bbox_inches.y0 - padding_subplots,
                bbox_inches.x1 + padding_subplots,
                bbox_inches.y1 + padding_subplots
            )

            # Save the subplot using the bounding box
            plt.savefig(os.path.join(self.path_barplot,
                                     f"barplot-groupby_{groupby}"
                                     f"-stack_{stack}-dataset_{dataset}.png"),
                        bbox_inches=bbox_inches_expanded)

        # Adjust the spacing after creating subplots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(
                os.path.join(self.path_barplot,
                             f"barplot-groupby_{groupby}"
                             f"-stack_{stack}-aggregate.png")
                )

        plt.show()

        return