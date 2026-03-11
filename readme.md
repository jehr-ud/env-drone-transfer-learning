# Multi-Agent Drone Navigation Environment

This repository contains a custom multi-agent reinforcement learning environment built on top of `gym-pybullet-drones`, designed for obstacle avoidance and goal-directed navigation using autonomous drones.

The project includes:

* Custom PyBullet drone environment
* Multi-agent obstacle navigation
* PPO training pipeline
* Visual goals and obstacles
* Extensible architecture for transfer learning and future SNN-based agents

---

## Project Structure

```text id="o2r9q1"
my_drone_transfer/
│
├── my_drone_transfer/
│   ├── envs/
│   │   └── multi_agent_obstacle_env.py
│   │
│   ├── agents/
│   │   └── ppo_agent.py
│   │
│   └── train/
│       └── train_ppo.py
│
├── models/
├── logs/
├── tests/
│
├── requirements.txt
├── setup.py
├── setup.sh
├── run_train.sh
└── README.md
```

---

## Requirements

* Python 3.10
* Conda (recommended)

---

## Quick Installation

The repository includes a setup script to install everything automatically.

Run:

```bash id="f4j7m2"
chmod +x setup.sh
./setup.sh
```

This script will:

* activate the Conda environment
* install all dependencies
* install the local package in editable mode

---

## Manual Installation

### 1. Create Conda environment

```bash id="t8v3s6"
conda create -n drone_transfer python=3.10 -y
conda activate drone_transfer
```

---

### 2. Install dependencies

```bash id="n5p8k1"
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Install local package

```bash id="r3h6x9"
pip install -e .
```

---

## Run Training

A training script is included for quick execution.

Run:

```bash id="k6u2w4"
chmod +x run_train.sh
./run_train.sh
```

This launches PPO training automatically.

---

## Manual Training

You can also run:

```bash id="v1y9d7"
python -m my_drone_transfer.train.train_ppo
```

---

## TensorBoard Monitoring

To visualize training:

```bash id="j4n8e5"
tensorboard --logdir logs
```

Then open:

```text id="m7s3a1"
http://localhost:6006
```

---

## Environment Features

The environment includes:

* Two autonomous drones
* Static walls
* Cubic obstacles
* Cylindrical obstacles
* Individual goals per drone
* Colored drone-goal assignment
* Reward shaping for:

  * goal progress
  * obstacle avoidance
  * collision avoidance
  * flight stabilization

---

## Testing Environment

Run:

```bash id="q2f5z8"
python tests/test_env.py
```

---

## Future Work

* Transfer Learning
* Curriculum Learning
* Spiking Neural Networks
* Dynamic obstacles
* Cooperative multi-agent policies

---

## Notes

This project uses `gym-pybullet-drones` as simulation base, while environment design, reward shaping, and learning setup are fully custom.
