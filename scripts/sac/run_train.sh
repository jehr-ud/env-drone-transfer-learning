#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone_transfer

echo "Launching training..."

export PYTHONPATH=$PYTHONPATH:$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE
python -m drone_transfer.train.train_sac