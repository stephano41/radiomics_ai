from autorad.training import Trainer as OrigTrainer
from src.utils.trainer import log_hydra



class Trainer(OrigTrainer):
    def log_to_mlflow(self, study):
        super().log_to_mlflow(study)

        log_hydra(self.result_dir)


