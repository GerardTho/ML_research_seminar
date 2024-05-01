#/bin/bash

python3 data/download_data.py
tensorboard --logdir=logs/runs --port=6006 & python3 main.py