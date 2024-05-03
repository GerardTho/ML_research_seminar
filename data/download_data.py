from data import create_dataset
import argparse
import yaml
import os
import s3fs
from dotenv import load_dotenv
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

load_dotenv()

if (
    not ("AWS_S3_ENDPOINT" in os.environ)
    or not ("AWS_ACCESS_KEY_ID" in os.environ)
    or not ("AWS_SECRET_ACCESS_KEY" in os.environ)
    or not ("AWS_SESSION_TOKEN" in os.environ)
):
    raise PermissionError(
        "No credentials in .env detected."
        + "Check if the .env file exist and is properly filled."
    )

logger.info("Credentials found")


def download_S3_folder(bucket, S3_directory, local_directory):
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": S3_ENDPOINT_URL},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"],
    )

    for dataset_path in fs.ls(f"{bucket}/{S3_directory}"):
        _, dataset_name = os.path.split(dataset_path)
        directory_path, folder_name = os.path.split(fs.ls(dataset_path)[0])
        logger.info(f"{dataset_name} downloading")
        outpath = f"{local_directory}/{dataset_name}/{folder_name}"
        os.makedirs(outpath, exist_ok=True)
        for file in tqdm(
            fs.ls(f"{directory_path}/{folder_name}"),
            desc=f"Downloading - {dataset_name} ",
        ):
            _, filename = os.path.split(file)
            if not (filename in os.listdir(outpath)):
                with fs.open(file) as f:
                    fs.download(
                        f, f"{local_directory}/{dataset_name}/{folder_name}"
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process a YAML configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="data/config.yml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    file_path = args.config
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    bucket = config.pop("bucket", "tgerard")
    S3_directory = config.pop("S3_directory", "diffusion/datasets/")
    local_directory = config.pop("local_directory", "data/datasets/")

    logger.info("Downloading data")
    download_S3_folder(bucket, S3_directory, local_directory)

    logger.info("Processing dataset")
    for dataset_name in os.listdir(local_directory):
        os.makedirs(
            os.path.join(local_directory, dataset_name, "processed"),
            exist_ok=True,
        )
        create_dataset(local_directory, dataset_name)
