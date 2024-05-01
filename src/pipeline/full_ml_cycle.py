import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
from autorad.models import MLClassifier
from omegaconf import OmegaConf

from src.preprocessing import run_auto_preprocessing
from src.training import Trainer
from src.pipeline import evaluate_run, analyse_run
from src.utils.pipeline import get_multimodal_feature_dataset, split_feature_dataset

logger = logging.getLogger(__name__)


def preprocess_data(config, feature_dataset, output_dir):
    if config.get('existing_processed_data_dir') is None:
        run_auto_preprocessing(data=feature_dataset.data,
                               result_dir=output_dir,
                               **OmegaConf.to_container(config.preprocessing, resolve=True))
        logger.info('Preprocessed finished')
        return output_dir

    logger.info(f'Using existing preprocessing directory at {config.get("existing_processed_data_dir")}')
    return Path(config.get('existing_processed_data_dir'))


def initialize_models(config):
    logger.info('Initialising models')
    if config.models is None:
        return MLClassifier.initialize_default_sklearn_models()
    return [MLClassifier.from_sklearn(model_name) for model_name in
            OmegaConf.to_container(config.models, resolve=True)]


def train_models(config, feature_dataset, train_output_dir, run_name):
    trainer = Trainer(
        dataset=feature_dataset,
        models=initialize_models(config),
        result_dir=train_output_dir,
        multi_class=config.multi_class,
        labels=config.labels,
        **config.get('trainer', {})
    )
    experiment_name = config.name or datetime.now().strftime('%Y%m%d%H%M%S')
    trainer.set_optimizer('optuna', n_trials=config.optimizer.n_trials)
    trainer.run(auto_preprocess=True, experiment_name=experiment_name,
                mlflow_start_kwargs=dict(description=config.get('notes', None),
                                         run_name=f'{run_name}'))
    logger.info('Hyperparameter tuning finished')


def full_ml_cycle(config):
    """
    Executes the full machine learning cycle which includes data setup, feature extraction,
    preprocessing, model training, evaluation, and analysis based on a provided configuration.

    Parameters:
        config (DictConfig): Configuration object containing all settings for dataset preparation,
                             model training, evaluation, and analysis.

    Steps:
    1. Set up and extract features from the dataset.
    2. Split the dataset into training and validation as per configuration.
    3. Optionally run preprocessing steps.
    4. Train models using the prepared dataset.
    5. Evaluate the models.
    6. Run additional analysis as defined in the pipeline.

    Outputs:
    - Extracted feature data and any preprocessing results are saved to disk.
    - Model evaluation results and any subsequent analysis are logged and can be further processed.

    Returns:
        None
    """
    # Initialize dataset and extract features based on configuration
    feature_dataset = get_multimodal_feature_dataset(**OmegaConf.to_container(config.feature_dataset, resolve=True))

    # Define the output directory based on the current hydra configuration run directory
    output_dir = os.path.normpath(hydra.utils.HydraConfig.get().run.dir)

    # Save the extracted feature data to CSV in the output directory
    feature_dataset.df.to_csv(os.path.join(output_dir, 'extracted_features.csv'))

    # Split the dataset according to configuration and save the split details
    feature_dataset = split_feature_dataset(feature_dataset,
                                            save_path=os.path.join(output_dir, 'splits.yml'),
                                            **config.split)

    # Process data and train models using the configured settings
    train_output_dir = preprocess_data(config, feature_dataset, output_dir)
    train_models(config, feature_dataset, train_output_dir, f'{output_dir.split("/")[-2]}@{output_dir.split("/")[-1]}')

    logger.info('Starting evaluation')
    evaluate_run(config)

    logger.info('Starting analysis pipeline')
    analyse_run(config)
