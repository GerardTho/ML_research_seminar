import yaml
import os
from training_script import train_model_from_config
import torch
from data.homophily.homophily_functions import get_homophily
from visualisation.plot import prepare_data, to_table, save_plot
from data.data import create_dataset
from data.utils import download_S3_folder

global_pooling_layer_to_test = ["mean", "max"]
local_pooling_layers_to_test = ["SAG","TOPK", None]
convolution_layers_to_test = ["GCN", "GAT", "GINConv"]

bucket="tgerard"
S3_directory="diffusion/data/datasets/"
local_directory="data/datasets/"

# Download data

download_S3_folder(bucket, S3_directory, local_directory)

# Define your configuration data

path_templates = "configs/templates"
path_generated = "configs/generated"
os.makedirs(path_generated, exist_ok=True)
configs_path = os.listdir(path_templates)

for config_path in configs_path:
    if config_path.endswith(".yml"):
        with open(os.path.join(path_templates,config_path), 'r') as config_file:
            config_model = yaml.safe_load(config_file)
    
        # Recreate the dataset from the graphs
        dataset_name = config_model["model"]["dataset"]
        dataset_path = config_model["model"]["dataset_path"]
        os.makedirs(os.path.join(dataset_path,dataset_name,"processed"), exist_ok=True)
        create_dataset(dataset_path, dataset_name)
        get_homophily('datasets' , dataset_name)

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

for config_path in configs_generated_path:
    if config_path.endswith(".yaml"):
        train_model_from_config(os.path.join(path_generated,config_path))

list_dict = prepare_data.get_list_dict()
plot = save_plot.Plot(list_dict)
plot.plot_all(train=True)
plot.plot_all(train=False)
table = to_table.ToTable(list_dict, per_dataset=True)
table.save_all()