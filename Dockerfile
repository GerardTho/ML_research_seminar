FROM ubuntu:22.04
WORKDIR ${HOME}
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY data ./data
COPY model ./model
COPY visualisation ./visualisation
COPY api ./api
COPY configs ./configs
COPY main.py /main.py
COPY install.sh /install.sh
RUN --mount=type=secret,id=AWS_ACCESS_KEY_ID \
  --mount=type=secret,id=AWS_S3_ENDPOINT \
  --mount=type=secret,id=AWS_SECRET_ACCESS_KEY \
  --mount=type=secret,id=AWS_SESSION_TOKEN \
   export AWS_ACCESS_KEY_ID=$(cat /run/secrets/AWS_ACCESS_KEY_ID) && \
   export AWS_S3_ENDPOINT=$(cat /run/secrets/AWS_S3_ENDPOINT) && \
   export AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/AWS_SECRET_ACCESS_KEY) && \
   export AWS_SESSION_TOKEN=$(cat /run/secrets/AWS_SESSION_TOKEN)  && \
   chmod +x /install.sh  && \ 
   /install.sh
CMD ["bash", "-c", "./api/run.sh"]