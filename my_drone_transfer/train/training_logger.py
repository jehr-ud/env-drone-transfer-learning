import os
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import pandas as pd


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.records = []
        
        # Crear la carpeta de modelos si no existe
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 1. Registro de métricas (lo que ya tenías)
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.records.append({
                "timesteps": self.num_timesteps,
                "reward": ep_info["r"],
                "length": ep_info["l"]
            })

        # 2. Guardado parcial (Checkpoint)
        if self.n_calls % self.save_freq == 0:
            path_model = os.path.join(self.save_path, f"model_step_{self.num_timesteps}")
            path_stats = os.path.join(self.save_path, f"stats_step_{self.num_timesteps}.pkl")
            
            # Guardar el modelo
            self.model.save(path_model)
            
            # Guardar las estadísticas del VecNormalize (OBLIGATORIO)
            if self.training_env is not None:
                self.training_env.save(path_stats)
            
            if self.verbose > 0:
                print(f"Checkpoint guardado en el paso {self.num_timesteps}")
                print(f"Modelo: {path_model}.zip | Stats: {path_stats}")

        return True

    def _on_training_end(self):
        df = pd.DataFrame(self.records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_metrics_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Training metrics saved: {filename}")