#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone_transfer

pip install --upgrade pip setuptools wheel

conda install -c conda-forge pybullet -y
conda install setuptools -y

pip install -r requirements.txt

pip install --no-deps git+https://github.com/utiasDSL/gym-pybullet-drones.git

pip install -e .

python -c "import pybullet, gymnasium, stable_baselines3, transforms3d; print('✅ Environment ready')"