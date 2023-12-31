import logging

import hydra
from autorad.inference.infer_utils import get_last_run_from_experiment_name, load_dataset_artifacts, \
    load_feature_dataset

from src.evaluation import log_ci2mlflow
from src.evaluation._bootstrap import Bootstrap
from src.utils.infer_utils import get_pipeline_from_last_run

logger = logging.getLogger(__name__)


def evaluate_last(config):
    last_run = get_last_run_from_experiment_name(config.name)
    logger.info(f'evaluating {last_run.run_id}')
    pipeline = get_pipeline_from_last_run(config.name)

    dataset_artifacts = load_dataset_artifacts(last_run)
    feature_dataset = load_feature_dataset(feature_df=dataset_artifacts['df'],
                                           dataset_config=dataset_artifacts['dataset_config'],
                                           splits=dataset_artifacts['splits'])

    evaluator = Bootstrap(feature_dataset.X, feature_dataset.y, log_dir=last_run.artifact_uri.replace('file:///','/'), **config.bootstrap)

    confidence_interval, tpr_fpr = evaluator.run(pipeline)

    logger.info(confidence_interval)
    log_ci2mlflow(confidence_interval, tpr_fpr,
                  run_id=last_run.run_id)

