FROM ubuntu:22.04
WORKDIR ${HOME}/gnn
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY data ./data
COPY model ./model
COPY visualisation .visualisation
COPY api ./api
CMD ["bash", "-c", "run.sh"]