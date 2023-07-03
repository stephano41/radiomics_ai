from functools import partial

from autorad.config.type_definitions import PathLike
from autorad.data import FeatureDataset, TrainingData
from autorad.models import MLClassifier
from autorad.training import Trainer as OrigTrainer, train_utils
import mlflow
import logging
from optuna.trial import Trial
import numpy as np
from typing import Sequence
from sklearn.metrics import roc_auc_score


log = logging.getLogger(__name__)


class Trainer(OrigTrainer):
    def __init__(self,
                 dataset: FeatureDataset,
                 models: Sequence[MLClassifier],
                 result_dir: PathLike,
                 multi_class: str = "raise",
                 labels = None,
                 average = "macro"
                 ):

        self.multi_class = multi_class
        # self.num_classes = num_classes

        self.auc_scorer = partial(roc_auc_score, average=average, multi_class=self.multi_class, labels=labels)
        super().__init__(dataset, models, result_dir)

    def get_auc(self, y_true, y_pred):
        try:
            auc = self.auc_scorer(y_true, y_pred)
        except ValueError as e:
            if 'Only one class present' not in str(e):
                raise ValueError(e)
            log.error("Only one class present in y_true. ROC AUC score is not defined in that case")
            auc = np.nan
        return auc

    # def run(
    #     self,
    #     auto_preprocess: bool = False,
    #     experiment_name="model_training",
    # ):
    #     """
    #            Run hyperparameter optimization for all the models.
    #            """
    #     mlflow.set_tracking_uri('file:/' + str(self.result_dir))
    #     if not mlflow.get_experiment_by_name(experiment_name):
    #         mlflow.create_experiment(experiment_name)
    #     else:
    #         log.warning("Running training in existing experiment.")
    #     mlflow.set_experiment(experiment_name)
    #     with mlflow.start_run():
    #         study = self.optimizer.create_study(
    #             study_name=experiment_name,
    #         )
    #
    #         study.optimize(
    #             lambda trial: self._objective(trial, auto_preprocess),
    #             n_trials=self.optimizer.n_trials,
    #         )
    #         best_trial = study.best_trial
    #         self.log_to_mlflow(
    #             best_trial=best_trial,
    #             auto_preprocess=auto_preprocess,
    #         )

    def _objective(self, trial: Trial, auto_preprocess=False) -> float:
        """Get params from optuna trial, return the metric."""
        data = self.get_trial_data(trial, auto_preprocess=auto_preprocess)

        model_name = trial.suggest_categorical(
            "model", [m.name for m in self.models]
        )
        model = train_utils.get_model_by_name(model_name, self.models)
        model = self.set_optuna_params(model, trial)
        aucs = []
        for (
            X_train,
            y_train,
            _,
            X_val,
            y_val,
            _,
        ) in data.iter_training():
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()
            X_val = X_val.to_numpy()
            y_val = y_val.to_numpy()

            try:
                model.fit(X_train, y_train)
            except ValueError:
                log.error(f"Training {model.name} failed.")
                return np.nan

            if self.multi_class == 'raise':
                y_pred = model.predict_proba_binary(X_val)
            elif self.multi_class == 'ovr' or self.multi_class == 'ovo':
                y_pred = model.predict_proba(X_val)
            else:
                raise ValueError(f"multi_class must be 'raise', 'ovr', 'ovo', got {self.multi_class}")

            try:
                auc_val = self.get_auc(y_val, y_pred)
            except ValueError as e:
                log.error(f"Evaluating {model.name} failed on auc. \n{e}")
                return np.nan

            aucs.append(auc_val)
        model.fit(
            data.X.train, data.y.train
        )  # refit on the whole training set (important for cross-validation)
        auc_val = float(np.nanmean(aucs))
        trial.set_user_attr("AUC_val", auc_val)
        trial.set_user_attr("model", model)
        trial.set_user_attr("data_preprocessed", data)

        return auc_val

    def log_train_auc(self, model: MLClassifier, data: TrainingData):
        y_true = data.y.train
        if self.multi_class == 'raise':
            y_pred_proba = model.predict_proba_binary(data.X.train)
        elif self.multi_class == 'ovr' or self.multi_class == 'ovo':
            y_pred_proba = model.predict_proba(data.X.train)
        else:
            raise ValueError(f"multi_class must be 'raise', 'ovr', 'ovo', got {self.multi_class}")

        train_auc = self.get_auc(y_true, y_pred_proba)

        mlflow.log_metric("AUC_train", float(train_auc))