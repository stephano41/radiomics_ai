from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any

import hydra.errors
import joblib
from autorad.config import config
from autorad.data import TrainingData
from autorad.preprocessing import Preprocessor as OrigPreprocessor
from autorad.preprocessing import run_auto_preprocessing as orig_auto_preprocessing
from hydra.utils import instantiate
from sklearn.compose import ColumnTransformer


log = logging.getLogger(__name__)


def run_auto_preprocessing(
        data: TrainingData,
        autoencoder=None,
        encoder_colname='ID',
        **kwargs 
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

    if autoencoder is not None:
        autoencoder_preprocessor = Preprocessor(standardize=False, feature_selection_method=None, oversampling_method=None, autoencoder=autoencoder,  encoder_colname=encoder_colname, feature_first=False)
        data = autoencoder_preprocessor.fit_transform_data(data)

    orig_auto_preprocessing(data=data, 
                            preprocessor_cls=Preprocessor,
                            autoencoder=autoencoder,
                            encoder_colname=encoder_colname,
                            **kwargs)


class Preprocessor(OrigPreprocessor):
    def __init__(self,
                 standardize: bool = True,
                 feature_selection_method: str | None = None,
                 oversampling_method: str | None = None,
                 random_state: int = config.SEED,
                 autoencoder=None,
                 encoder_colname='ID',
                 feature_first=True
                 ):
        self.autoencoder = autoencoder
        self.encoder_colname = encoder_colname
        super().__init__(standardize=standardize,
                         feature_selection_method=feature_selection_method,
                         oversampling_method=oversampling_method,
                         random_state=random_state,
                         feature_first=feature_first)


    def _build_pipeline(self):
        pipeline = super()._build_pipeline()
        if self.autoencoder is not None:
            autoencoder = instantiate_autoencoder(self.autoencoder)

            pipeline.steps.insert(0,
                ("autoencoder",
                 ColumnTransformer(transformers=[('autoencoder', autoencoder, self.encoder_colname)],
                                   remainder='passthrough',
                                   verbose_feature_names_out=False).set_output(transform='pandas'))
            )

        return pipeline


def instantiate_autoencoder(autoencoder_config):
    try:
        autoencoder = instantiate(autoencoder_config, _convert_='object')
    except hydra.errors.InstantiationException:
        if isinstance(autoencoder_config, collections.abc.Mapping):
            raise TypeError("autoencoder passed couldn't be instantiated but is a dictionary like object")
        print("hydra instantiation of autoencoder failed, autoencoder better be a working object")
        autoencoder = autoencoder_config
    return autoencoder
    

