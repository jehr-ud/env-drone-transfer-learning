# Training
#Base:
# PPO Scratch → Direct training in an obstacle course environment
# PPO Curriculum:
    #Phase 1 → Phase 2 → Phase 3
# Plastic Transfer:
    # Learn skills in simple environments
    # Transfer to an obstacle course environment


#echo "Evaluation PPO (1/3)"
#sh scripts/ppo/run_train.sh
#sh scripts/ppo/run_test.sh
#echo "Evaluation PPO (1/3) 👌"

echo "Evaluation PPO Curriculum (2/3)"
sh scripts/ppo/run_train_curriculum.sh
sh scripts/ppo/run_test_curriculum.sh
echo "Evaluation PPO Curriculum (2/3) 👌"

#echo "Evaluation Plastic-transfer(3/3)"
#sh scripts/plastic-transfer/run_train.sh
#sh scripts/plastic-transfer/run_test.sh
#echo "Evaluation Plastic-transfer(3/3) 👌"