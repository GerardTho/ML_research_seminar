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
COPY .env .env
# RUN /bin/bash -c /install.sh
CMD ["bash", "-c", "./api/run.sh"]