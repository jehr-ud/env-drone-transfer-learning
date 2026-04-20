import os
import csv
from datetime import datetime

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLoggerCallback(BaseCallback):
    def __init__(
            self,
            save_freq: int,
            save_path: str,
            name_algo: str,
            verbose=1
        ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        
        self.records = []
        self.csv_file = None
        self.csv_writer = None
        self.name_algo = name_algo

        os.makedirs(save_path, exist_ok=True)

    def _on_training_start(self):
        # CSV en vivo (mejor que guardar todo en RAM)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"training_steps_{timestamp}_{self.name_algo}.csv"

        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "reward"])

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")

        if rewards is not None:
            reward_value = float(rewards.mean())

            self.csv_writer.writerow([
                self.num_timesteps,
                reward_value
            ])

        # -------------------------------
        # CHECKPOINT
        # -------------------------------
        if self.n_calls % self.save_freq == 0:
            path_model = os.path.join(
                self.save_path,
                f"model_step_{self.num_timesteps}"
            )

            self.model.save(path_model)

            if self.verbose > 0:
                print(f"Checkpoint en step {self.num_timesteps}")

        return True

    def _on_training_end(self):

        if self.csv_file:
            self.csv_file.close()

        print(f"Training CSV saved at: {self.csv_path}")