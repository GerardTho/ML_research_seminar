#/bin/bash

# python data/download_data.py
tensorboard --logdir=logs/runs & python main.py && /bin/bash api/run.sh