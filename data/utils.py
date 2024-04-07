import os
import s3fs

def download_S3_folder(bucket, S3_directory, local_directory):
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
    fs.download(os.path.join(bucket, S3_directory), local_directory, recursive=True)