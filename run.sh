#/bin/bash

python3 data/download_data.py
tensorboard --logdir=logs/runs & python3 main.py && /bin/bash api/run.sh