import logging
from functools import partial
from typing import Sequence

import joblib
import mlflow
import numpy as np
from autorad.config.type_definitions import PathLike
from autorad.data import FeatureDataset, TrainingData
from autorad.models import MLClassifier
from autorad.training import Trainer as OrigTrainer, train_utils
from autorad.utils import mlflow_utils
from optuna.trial import Trial

from src.metrics import roc_auc, pr_auc
from src.preprocessing import Preprocessor

log = logging.getLogger(__name__)


class Trainer(OrigTrainer):
    def __init__(self,
                 dataset: FeatureDataset,
                 models: Sequence[MLClassifier],
                 result_dir: PathLike,
                 multi_class: str = "raise",
                 labels=None,
                 average="macro",
                 metric='roc_auc'
                 ):

        self.multi_class = multi_class
        # self.num_classes = num_classes

        # self.auc_scorer = partial(roc_auc_score, average=average, multi_class=self.multi_class, labels=labels)
        if metric=='roc_auc':
            self.get_auc = partial(roc_auc, average=average, multi_class=multi_class, labels=labels)
        elif metric=='pr_auc':
            self.get_auc = partial(pr_auc, average=average, multi_class=multi_class, labels=labels)
        else:
            raise ValueError(f'metric not implemented, got {metric}')

        self._existing_preprocess_kwargs = None
        super().__init__(dataset, models, result_dir)

    def run(
            self,
            auto_preprocess: bool = False,
            experiment_name="model_training",
    ):
        if auto_preprocess:
            _, self._existing_preprocess_kwargs = self.get_preprocessed_pickle()
        super().run(auto_preprocess, experiment_name)

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
            except ValueError as e:
                log.error(f"Training {model.name} failed. \n{e}")
                return np.nan

            y_pred = model.predict_proba(X_val)

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

    def get_best_preprocessed_dataset(self, trial: Trial) -> TrainingData:
        """ "
        Get preprocessed dataset with preprocessing method that performed
        best in the training.
        """
        preprocessed, _ = self.get_preprocessed_pickle()
        feature_selection_method = trial.suggest_categorical(
            "feature_selection_method", preprocessed.keys()
        )
        oversampling_method = trial.suggest_categorical(
            "oversampling_method",
            preprocessed[feature_selection_method].keys(),
        )
        result = preprocessed[feature_selection_method][oversampling_method]

        return result

    def get_preprocessed_pickle(self):
        pkl_path = self.result_dir / "preprocessed.pkl"
        with open(pkl_path, "rb") as f:
            preprocessed_data, preprocesser_kwargs = joblib.load(f)
        return preprocessed_data, preprocesser_kwargs

    def log_train_auc(self, model: MLClassifier, data: TrainingData):
        y_true = data.y.train

        y_pred_proba = model.predict_proba(data.X.train)

        train_auc = self.get_auc(y_true, y_pred_proba)
        print(mlflow.get_tracking_uri())
        mlflow.log_metric("AUC_train", float(train_auc))

    def save_best_preprocessor(self, best_trial_params: dict):
        feature_selection = best_trial_params["feature_selection_method"]
        oversampling = best_trial_params["oversampling_method"]
        preprocessor_kwargs = self._existing_preprocess_kwargs.copy()

        preprocessor_kwargs.update({
            'feature_selection_method': feature_selection,
            'oversampling_method': oversampling
        })
        preprocessor = Preprocessor(**preprocessor_kwargs)
        preprocessor.fit_transform_data(self.dataset.data)
        if "select" in preprocessor.pipeline.named_steps:
            selected_features = preprocessor.pipeline[
                "select"
            ].selected_features
            mlflow_utils.log_dict_as_artifact(
                selected_features, "selected_features"
            )
        if "autoencoder" in preprocessor.pipeline.named_steps:
            mlflow_utils.log_dict_as_artifact(
                preprocessor_kwargs['autoencoder'], "autoencoder"
            )

        mlflow.sklearn.log_model(preprocessor, "preprocessor")
