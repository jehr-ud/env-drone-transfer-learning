#!/bin/bash

echo "Activating conda environment..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone_transfer

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

echo "Running PPO test..."

python -m my_drone_transfer.tests.test_ppo