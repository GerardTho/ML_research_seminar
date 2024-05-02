import yaml
import os
from model.training_script import train_model_from_config
import torch
from data.homophily.homophily_functions import get_homophily
from visualisation.plot import prepare_data, to_table, save_plot, visualisation_plot
import logging
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a YAML configuration file and output the results to a JSON file.')
    parser.add_argument('-c', '--config', default="configs/config.yml", help='Path to the YAML configuration file')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename='logs/training.log', encoding='utf-8', level=logging.INFO)

    file_path=args.config

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    global_pooling_layer_to_test = config.pop("global_pooling_layer_to_test", ["mean", "max"])
    local_pooling_layers_to_test = config.pop("local_pooling_layers_to_test", ["SAG","TOPK"])
    local_pooling_layers_to_test.append(None)
    convolution_layers_to_test = config.pop("convolution_layers_to_test", ["GCN", "GAT", "GINConv"])

    # Define your configuration data

    path_templates = "configs/templates"
    path_generated = "configs/generated"
    os.makedirs(path_generated, exist_ok=True)
    configs_path = os.listdir(path_templates)

    for config_path in configs_path:
        if config_path.endswith(".yml"):
            with open(os.path.join(path_templates, config_path), 'r') as config_file:
                config_model = yaml.safe_load(config_file)
        
            # Recreate the dataset from the graphs
            dataset_name = config_model["model"]["dataset"]
            dataset_path = config_model["model"]["dataset_path"]
            get_homophily('datasets', dataset_name)

            for convolution_layer in convolution_layers_to_test:
                config_model["model"]["convolution_layer"] = convolution_layer
                for global_pooling_layer in global_pooling_layer_to_test:
                    config_model["model"]["global_pooling_layer"] = global_pooling_layer
                    for local_pooling_layer in local_pooling_layers_to_test:
                        config_model["model"]["local_pooling_layer"] = local_pooling_layer
                        # Specify the file path where you want to save the YAML configuration
                        config_name = f"{dataset_name}_{convolution_layer}_{global_pooling_layer}_{local_pooling_layer}.yaml"
                        config_model["model"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        
                        # Write the configuration data to the YAML file
                        with open(os.path.join(path_generated,config_name), 'w') as config_file:
                            yaml.dump(config_model, config_file)


    configs_generated_path = os.listdir(path_generated)

    # Train the models
    for config_path in configs_generated_path:
        if config_path.endswith(".yaml"):
            train_model_from_config(os.path.join(path_generated,config_path))

    # Save the results
    list_dict = prepare_data.get_list_dict()
    plot = save_plot.Plot(list_dict)
    plot.plot_all(train=True)
    plot.plot_all(train=False)
    vplot = visualisation_plot.VisualisationPlot(list_dict)
    vplot.save()
    table = to_table.ToTable(list_dict, per_dataset=True)
    table.save_all()