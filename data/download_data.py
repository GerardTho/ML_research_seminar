from data import create_dataset
import argparse
import yaml
import os
import s3fs
from dotenv import load_dotenv

load_dotenv()

def download_S3_folder(bucket, S3_directory, local_directory):
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL}, 
                            key = os.environ["AWS_ACCESS_KEY_ID"], 
                            secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
                            token = os.environ["AWS_SESSION_TOKEN"])    
    fs.download(f'{bucket}/{S3_directory}', local_directory, recursive=True)

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