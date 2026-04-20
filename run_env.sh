#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone_transfer

echo "Run single env..."
python -m drone_transfer.tests.test_env