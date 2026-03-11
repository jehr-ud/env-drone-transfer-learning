#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone_transfer

python -m my_drone_transfer.train.train_ppo