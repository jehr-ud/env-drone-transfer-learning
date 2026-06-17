# Training
#Base:
# PPO Scratch → Direct training in an obstacle course environment
# PPO Curriculum:
    #Phase 1 → Phase 2 → Phase 3
# Plastic Transfer:
    # Learn skills in simple environments
    # Transfer to an obstacle course environment

echo "Cleaning old models and logs.."

#!/bin/bash

read -p "Are you sure you want to delete 'models' and 'logs'? (y/n): " answer

if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    rm -rf models
    rm -rf logs
    echo "Directories deleted."
else
    echo "Operation cancelled."
fi

mkdir logs

echo "Train Plastic-transfer(1/3)"
sh scripts/plastic-transfer/run_train.sh
sh scripts/plastic-transfer/run_test.sh
echo "Train Plastic-transfer(1/3) 👌"

echo "Train and Test PPO (2/3)"
sh scripts/ppo/run_train.sh
sh scripts/ppo/run_test.sh
echo "Train and Test PPO (2/3) 👌"

echo "Train PPO Curriculum (3/3)"
sh scripts/ppo/run_train_curriculum.sh
sh scripts/ppo/run_test_curriculum.sh
echo "Train PPO Curriculum (3/3) 👌"
