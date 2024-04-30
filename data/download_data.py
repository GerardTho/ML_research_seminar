from utils import download_S3_folder
from data import create_dataset
import argparse
import yaml
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process a YAML configuration file and output the results to a JSON file.')
    parser.add_argument('-c', '--config', default="data/config.yml", help='Path to the YAML configuration file')
    args = parser.parse_args()

    file_path = args.config
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    bucket = config.pop("bucket", "tgerard")
    S3_directory = config.pop("S3_directory", "diffusion/datasets/")
    local_directory = config.pop("local_directory", "data/datasets/")
    download_S3_folder(bucket, S3_directory, local_directory)
    
    for dataset_name in os.listdir(local_directory):
        os.makedirs(os.path.join(local_directory, dataset_name, "processed"), exist_ok=True)
        create_dataset(local_directory, dataset_name)