import logging

from autorad.inference.infer_utils import get_last_run_from_experiment_name, load_dataset_artifacts, \
    load_feature_dataset

from src.evaluation import log_ci2mlflow, Bootstrap
from autorad.inference.infer_utils import get_pipeline_from_run, get_run_info_as_series

logger = logging.getLogger(__name__)


def evaluate_run(config):
    if config.get('run_id', None) is not None:
        # convert this to the same format
        run = get_run_info_as_series(config.run_id)
    else:
        logger.info(f'no run speicified in config, getting the last run from {config.name} instead')
        run = get_last_run_from_experiment_name(config.name)

    logger.info(f'evaluating {run.run_id}')
    pipeline = get_pipeline_from_run(run)

    dataset_artifacts = load_dataset_artifacts(run)
    feature_dataset = load_feature_dataset(feature_df=dataset_artifacts['df'],
                                           dataset_config=dataset_artifacts['dataset_config'],
                                           splits=dataset_artifacts['splits'])

    evaluator = Bootstrap(feature_dataset.X, feature_dataset.y, log_dir=run.artifact_uri.replace('file:///', '/'),
                          **config.bootstrap)

    confidence_interval, tpr_fpr, preds = evaluator.run(pipeline)

    logger.info(confidence_interval)
    log_ci2mlflow(confidence_interval, tpr_fpr, run_id=run.run_id)
