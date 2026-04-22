#!/bin/bash

set -e

echo "🔄 Activating conda env..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone_transfer

# -------------------------------
# Clean broken setuptools (fix distutils bug)
# -------------------------------
echo "🧹 Cleaning broken setuptools..."

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

rm -f $SITE_PACKAGES/distutils-precedence.pth || true

# -------------------------------
# Core tools (ONLY pip)
# -------------------------------
echo "⬆️ Upgrading core tools..."
pip install --upgrade pip wheel
pip install --force-reinstall setuptools

# -------------------------------
# System deps (conda only where needed)
# -------------------------------
echo "📦 Installing system dependencies..."
conda install -c conda-forge pybullet -y

# -------------------------------
# Install requirements
# -------------------------------
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# -------------------------------
# RLlib (MAPPO)
# -------------------------------
echo "🤖 Installing RLlib..."
pip install "ray[rllib]"

# -------------------------------
# Gym PyBullet Drones
# -------------------------------
echo "🚁 Installing gym-pybullet-drones..."
pip install --no-deps git+https://github.com/utiasDSL/gym-pybullet-drones.git

# -------------------------------
# Packages
# -------------------------------
git clone https://github.com/jehr-ud/plastic-transfer.git plastic-transfer-base
pip install -e plastic-transfer-base
pip install -e .

# -------------------------------
# Fix TensorBoard (important)
# -------------------------------
echo "📊 Fixing TensorBoard..."
pip install --upgrade tensorboard

# -------------------------------
# Final test
# -------------------------------
echo "🧪 Testing environment..."

python - <<EOF
import pkg_resources
import pybullet
import gymnasium
import ray
from ray.rllib.algorithms.ppo import PPOConfig

print("✅ Environment ready with RLlib + MAPPO")
EOF