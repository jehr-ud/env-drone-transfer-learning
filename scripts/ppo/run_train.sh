#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone_transfer

echo "Cleaning up old TensorBoard processes..."
kill -9 $(lsof -t -i :6006) 2>/dev/null || true

echo "Starting TensorBoard..."
python -m tensorboard.main --logdir ./ppo_drone_tensorboard/ --port 6006 &

sleep 3
open http://localhost:6006

echo "Launching training..."

export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m drone_transfer.train.train_ppo