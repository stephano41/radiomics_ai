from __future__ import annotations

import inspect
from typing import Any

from autorad.feature_selection import create_feature_selector
from autorad.preprocessing import Preprocessor as OrigPreprocessor, oversample_utils
import logging
from pathlib import Path

import joblib

from autorad.config import config
from autorad.data import TrainingData, TrainingInput, TrainingLabels
from hydra.utils import instantiate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import FeatureDataset

log = logging.getLogger(__name__)


def run_auto_preprocessing(
        data: TrainingData,
        result_dir: Path,
        use_oversampling: bool = True,
        use_feature_selection: bool = True,
        oversampling_methods: list[str] | None = None,
        feature_selection_methods: list[str] | None = None,
        autoencoder=None,
        encoder_colname='ID'
):
    """Run preprocessing with a variety of feature selection and oversampling methods.

    Args:
    - data: Training data to preprocess.
    - result_dir: Path to a directory where the preprocessed data will be saved.
    - use_oversampling: A boolean indicating whether to use oversampling. If `True` and
      `oversampling_methods` is not provided, all methods in the `config.OVERSAMPLING_METHODS`
      list will be used.
    - use_feature_selection: A boolean indicating whether to use feature selection. If `True` and
      `feature_selection_methods` is not provided, all methods in the `config.FEATURE_SELECTION_METHODS`
    - oversampling_methods: A list of oversampling methods to use. If not provided, all methods
      in the `config.OVERSAMPLING_METHODS` list will be used.
    - feature_selection_methods: A list of feature selection methods to use. If not provided, all
      methods in the `config.FEATURE_SELECTION_METHODS` list will be used.

    Returns:
    - None. The preprocessed data will be saved to the `result_dir` directory.
    """
    if use_oversampling:
        if oversampling_methods is None:
            oversampling_methods = config.OVERSAMPLING_METHODS
    else:
        oversampling_methods = [None]

    if use_feature_selection:
        if feature_selection_methods is None:
            feature_selection_methods = config.FEATURE_SELECTION_METHODS
    else:
        feature_selection_methods = [None]

    preprocessed = {}
    for selection_method in feature_selection_methods:
        preprocessed[selection_method] = {}
        for oversampling_method in oversampling_methods:
            preprocessor = Preprocessor(
                standardize=True,
                feature_selection_method=selection_method,
                oversampling_method=oversampling_method,
                autoencoder=autoencoder,
                encoder_colname=encoder_colname
            )
            try:
                preprocessed[selection_method][
                    oversampling_method
                ] = preprocessor.fit_transform_data(data)
            except AssertionError:
                log.error(
                    f"Preprocessing failed with {selection_method} and {oversampling_method}."
                )
        if not preprocessed[selection_method]:
            del preprocessed[selection_method]
    with open(Path(result_dir) / "preprocessed.pkl", "wb") as f:
        joblib.dump((preprocessed, preprocessor.get_params()), f)


class Preprocessor(OrigPreprocessor):
    def __init__(self,
                 standardize: bool = True,
                 feature_selection_method: str | None = None,
                 oversampling_method: str | None = None,
                 feature_selection_kwargs: dict[str, Any] | None = None,
                 random_state: int = config.SEED,
                 autoencoder=None,
                 encoder_colname='ID'
                 ):
        self.autoencoder = autoencoder
        self.encoder_colname = encoder_colname
        super().__init__(standardize=standardize,
                         feature_selection_method=feature_selection_method,
                         oversampling_method=oversampling_method,
                         feature_selection_kwargs=feature_selection_kwargs,
                         random_state=random_state)

    def fit_transform(
            self, X: TrainingInput, y: TrainingLabels
    ) -> tuple[TrainingInput, TrainingLabels]:

        result_X = {}
        result_y = {}
        transformed = self.pipeline.fit_transform(X.train, y.train)
        if isinstance(transformed, tuple):
            result_X["train"], result_y["train"] = transformed
        else:
            result_X["train"] = transformed
            result_y["train"] = y.train
        # allow for empty test set
        if not X.test.empty:
            result_X["test"] = self.pipeline.transform(X.test)
            result_y["test"] = y.test
        else:
            result_X["test"] = None
            result_y["test"] = None

        if X.val is not None:
            result_X["val"] = self.pipeline.transform(X.val)
            result_y["val"] = y.val
        if X.train_folds is not None and X.val_folds is not None:
            (
                result_X["train_folds"],
                result_y["train_folds"],
                result_X["val_folds"],
                result_y["val_folds"],
            ) = self._fit_transform_cv_folds(X, y)
        X_preprocessed = TrainingInput(**result_X)
        y_preprocessed = TrainingLabels(**result_y)
        return X_preprocessed, y_preprocessed

    def transform(self, X: TrainingInput):
        result_X = {}
        result_X["train"] = self.pipeline.transform(X.train)
        # allow for empty test set
        if not X.test.empty:
            result_X["test"] = self.pipeline.transform(X.test)
        else:
            result_X["test"] = None

        if X.val is not None:
            result_X["val"] = self.pipeline.transform(X.val)
        if X.train_folds is not None and X.val_folds is not None:
            (
                result_X["train_folds"],
                result_X["val_folds"],
            ) = self._transform_cv_folds(X)
        X_preprocessed = TrainingInput(**result_X)
        return X_preprocessed

    def _build_pipeline(self):
        steps = []
        if self.autoencoder is not None:
            steps.append(
                ("autoencoder",
                 ColumnTransformer(transformers=[('autoencoder', instantiate(self.autoencoder), self.encoder_colname)],
                                   remainder='passthrough',
                                   verbose_feature_names_out=False).set_output(transform='pandas'))
            )

        if self.standardize:
            steps.append(
                (
                    "standardize",
                    StandardScaler().set_output(transform="pandas"),
                )
            )
        if self.feature_selection_method is not None:
            steps.append(
                (
                    "select",
                    create_feature_selector(
                        method=self.feature_selection_method,
                        **self.feature_selection_kwargs,
                    ),
                ),
            )
        if self.oversampling_method is not None:
            steps.append(
                (
                    "oversample",
                    oversample_utils.OversamplerWrapper(
                        oversample_utils.create_oversampling_model(
                            method=self.oversampling_method,
                            random_state=self.random_state,
                        )
                    ),
                )
            )
        pipeline = Pipeline(steps)
        return pipeline

    def get_params(self, deep=None):
        return {key: getattr(self, key) for key in inspect.signature(Preprocessor.__init__).parameters.keys() if
                key != "self"}
