import hydra
from autorad.inference.infer_utils import get_last_run_from_experiment_name, load_dataset_artifacts, \
    load_feature_dataset

from src.evaluation import bootstrap, log_ci2mlflow
from src.utils.infer_utils import get_pipeline_from_last_run
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='main', version_base='1.3')
def main(config):
    last_run = get_last_run_from_experiment_name(config.name)
    logger.info(f'evaluating {last_run.run_id}')
    pipeline = get_pipeline_from_last_run(config.name)

    dataset_artifacts = load_dataset_artifacts(last_run)
    feature_dataset = load_feature_dataset(feature_df=dataset_artifacts['df'],
                                           dataset_config=dataset_artifacts['dataset_config'],
                                           splits=dataset_artifacts['splits'])

    confidence_interval, raw_scores = bootstrap(pipeline, feature_dataset.X, feature_dataset.y,
                                    **config.bootstrap)

    logger.info(confidence_interval)
    log_ci2mlflow(confidence_interval, raw_scores=raw_scores,
                  run_id=last_run.run_id)


if __name__ == '__main__':
    main()
