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
from optuna.trial import Trial
import multiprocessing as mp
from src.metrics import roc_auc, pr_auc
from src.utils.trainer import log_preprocessed_data, log_hydra
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
                 metric='roc_auc',
                 n_jobs=1
                 ):

        self.multi_class = multi_class
        self.n_jobs = n_jobs

        # self.auc_scorer = partial(roc_auc_score, average=average, multi_class=self.multi_class, labels=labels)
        if metric == 'roc_auc':
            self.get_auc = partial(roc_auc, average=average, multi_class=multi_class, labels=labels)
        elif metric == 'pr_auc':
            self.get_auc = partial(pr_auc, average=average, multi_class=multi_class, labels=labels)
        else:
            raise ValueError(f'metric not implemented, got {metric}')

        super().__init__(dataset, models, result_dir)


    def _objective(self, trial: Trial, auto_preprocess=False) -> float:
        """Get params from optuna trial, return the metric."""
        data = self.get_trial_data(trial, auto_preprocess=auto_preprocess)

        model_name = trial.suggest_categorical(
            "model", [m.name for m in self.models]
        )
        model = train_utils.get_model_by_name(model_name, self.models)
        model = self.set_optuna_params(model, trial)
        aucs = []
        try:
            if len(data.X.train_folds) > self.n_jobs:
                mp.set_start_method('spawn', force=True)
                with mp.Pool(processes=self.n_jobs) as pool:
                    for auc in pool.map(self._fit_and_evaluate, [(model, X_train, y_train, X_val, y_val) for
                                                              X_train, y_train, _, X_val, y_val, _ in
                                                              data.iter_training()]):
                        aucs.append(auc)

            else:
                for X_train, y_train, _, X_val, y_val, _ in data.iter_training():
                    auc_val = self._fit_and_evaluate((model, X_train, y_train, X_val, y_val))
                    aucs.append(auc_val)
        except Exception as e:
            log.warning(f"training {trial.params} failed")
            raise e

        auc_val = float(np.nanmean(aucs))
        trial.set_user_attr("AUC_val", auc_val)
        trial.set_user_attr("std_AUC_val", float(np.nanstd(aucs)))
        trial.set_user_attr("model", model)
        trial.set_user_attr("data_preprocessed", data)

        return auc_val

    def _fit_and_evaluate(self, args):
        """Fit the model and evaluate on validation data."""
        model, X_train, y_train, X_val, y_val = args

        _X_train = X_train.copy()
        _y_train = y_train.copy()

        model.fit(_X_train, _y_train)

        y_pred = model.predict_proba(X_val)

        auc_val = self.get_auc(y_val, y_pred)

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
            preprocessed_data, preprocessor_kwargs = joblib.load(f)
        return preprocessed_data, preprocessor_kwargs

    def log_train_auc(self, model: MLClassifier, data: TrainingData):
        y_true = data.y.train
        X_train = data.X.train

        model.fit(X_train, y_true)

        y_pred_proba = model.predict_proba(X_train)

        train_auc = self.get_auc(y_true, y_pred_proba)
        mlflow.log_metric("AUC_train", float(train_auc))

    def save_best_preprocessor(self, best_trial_params: dict):
        feature_selection = best_trial_params["feature_selection_method"]
        oversampling = best_trial_params["oversampling_method"]

        _, preprocessor_kwargs = self.get_preprocessed_pickle()

        preprocessor_kwargs.update({
            'feature_selection_method': feature_selection,
            'oversampling_method': oversampling
        })
        preprocessor = Preprocessor(**preprocessor_kwargs)
        preprocessor._fit_transform(self.dataset.X, self.dataset.y)

        mlflow.sklearn.log_model(preprocessor, "preprocessor")

    def get_best_trial(self, study):
        study_df = study.trials_dataframe(attrs=('number', 'user_attrs'))
        unique_auc = np.unique(study_df['user_attrs_AUC_val'])
        cutoff = unique_auc[-int(len(unique_auc) * 0.025)]

        selected_trials = study_df[study_df['user_attrs_AUC_val'] >= cutoff]

        min_std_row = selected_trials[
            selected_trials['user_attrs_std_AUC_val'] == selected_trials['user_attrs_std_AUC_val'].min()]

        best_trial_number = min_std_row['number'].iloc[0]

        for trial in study.trials:
            if trial.number == best_trial_number:
                log.info(
                    f'Best trial was number {best_trial_number}, {trial.params} with AUC: {trial.user_attrs["AUC_val"]} and standard deviation: {trial.user_attrs["std_AUC_val"]} ')
                return trial
        raise ValueError("trial number of best trial not found in study dataframe!")

    def log_to_mlflow(self, study):
        best_trial = self.get_best_trial(study)
        best_auc = best_trial.user_attrs["AUC_val"]
        mlflow.log_metric("AUC_val", best_auc)

        train_utils.log_optuna(study)

        best_model = best_trial.user_attrs["model"]
        best_model.fit(self.dataset.X, self.dataset.y)
        best_model.save_to_mlflow()

        best_params = best_trial.params
        self.save_params(best_params)
        self.save_best_preprocessor(best_params)
        self.copy_extraction_artifacts()
        train_utils.log_dataset(self.dataset)
        train_utils.log_splits(self.dataset.splits)

        data_preprocessed = best_trial.user_attrs["data_preprocessed"]
        self.log_train_auc(best_model, data_preprocessed)
        log_preprocessed_data(data_preprocessed)

        log_hydra(self.result_dir)


