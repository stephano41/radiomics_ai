import logging
import pandas as pd
import hydra
from autorad.inference.infer_utils import get_last_run_from_experiment_name, load_dataset_artifacts, \
    load_feature_dataset

from src.evaluation import log_ci2mlflow
from src.evaluation._bootstrap import Bootstrap
import mlflow
from autorad.models.classifier import MLClassifier

logger = logging.getLogger(__name__)


def evaluate_run(config):
    if config.get('run_id', None) is not None:
        # convert this to the same format
        run = pd.Series(dict(mlflow.get_run(config.run_id).info))
    else:
        logger.info(f'no run speicified in config, getting the last run from {config.name} instead')
        run = get_last_run_from_experiment_name(config.name)

    logger.info(f'evaluating {run.run_id}')
    model = MLClassifier.load_from_mlflow(f"{run['artifact_uri']}/model")
    preprocessor = mlflow.sklearn.load_model(f"{run['artifact_uri']}/preprocessor")
    pipeline = preprocessor.pipeline
    pipeline.steps.append(['estimator', model])

    dataset_artifacts = load_dataset_artifacts(run)
    feature_dataset = load_feature_dataset(feature_df=dataset_artifacts['df'],
                                           dataset_config=dataset_artifacts['dataset_config'],
                                           splits=dataset_artifacts['splits'])

    evaluator = Bootstrap(feature_dataset.X, feature_dataset.y, log_dir=run.artifact_uri.replace('file:///','/'), **config.bootstrap)

    confidence_interval, tpr_fpr, preds = evaluator.run(pipeline)

    logger.info(confidence_interval)
    log_ci2mlflow(confidence_interval, tpr_fpr,
                  run_id=run.run_id)

