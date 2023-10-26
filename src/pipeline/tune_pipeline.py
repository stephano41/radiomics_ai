import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
from autorad.models import MLClassifier
from omegaconf import OmegaConf

from src.preprocessing import run_auto_preprocessing
from src.training import Trainer
from .evaluate_last import evaluate_last
from .pipeline_components import get_multimodal_feature_dataset, split_feature_dataset

logger = logging.getLogger(__name__)


def tune_pipeline(config):
    # setup dataset, extract features, split the data
    feature_dataset = get_multimodal_feature_dataset(**OmegaConf.to_container(config.feature_dataset, resolve=True))

    output_dir = hydra.utils.HydraConfig.get().run.dir
    # save the feature_dataset
    feature_dataset.df.to_csv(os.path.join(output_dir, 'extracted_features.csv'))

    feature_dataset = split_feature_dataset(feature_dataset,
                                            save_path=os.path.join(output_dir, 'splits.yml'),
                                            **config.split)

    # initialise models
    if config.models is None:
        models = MLClassifier.initialize_default_sklearn_models()
    else:
        models = [MLClassifier.from_sklearn(model_name) for model_name in config.models]

    # run auto preprocessing
    run_auto_preprocessing(data=feature_dataset.data,
                           result_dir=Path(output_dir),
                           **OmegaConf.to_container(config.preprocessing, resolve=True))

    # start training
    trainer = Trainer(
        dataset=feature_dataset,
        models=models,
        result_dir=output_dir,
        multi_class=config.multi_class,
        labels=config.labels
    )
    if config.name is None:
        experiment_name = datetime.now().strftime('%Y%m%d%H%M%S')
    else:
        experiment_name = config.name

    trainer.set_optimizer('optuna', n_trials=config.optimizer.n_trials)
    trainer.run(auto_preprocess=True, experiment_name=experiment_name)

    # start evaluation
    evaluate_last(config)

# TODO calibration score?
