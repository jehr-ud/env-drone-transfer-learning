from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd


class TrainingLoggerCallback(BaseCallback):

    def __init__(self):
        super().__init__()

        self.records = []

    def _on_step(self) -> bool:

        if len(self.model.ep_info_buffer) > 0:

            ep_info = self.model.ep_info_buffer[-1]

            self.records.append({
                "timesteps": self.num_timesteps,
                "reward": ep_info["r"],
                "length": ep_info["l"]
            })

        return True

    def _on_training_end(self):

        df = pd.DataFrame(self.records)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"training_metrics_{timestamp}.csv"

        df.to_csv(filename, index=False)

        print(f"Training metrics saved: {filename}")