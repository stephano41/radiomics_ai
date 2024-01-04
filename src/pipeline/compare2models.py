import os

import hydra
import mlflow
from autorad.data import FeatureDataset
from autorad.inference.infer_utils import load_dataset_artifacts
from src.utils.infer_utils import get_pipeline_from_run
from src.evaluation.f_test import combined_ftest_5x2cv
import logging

logger = logging.getLogger(__name__)

def compare2models(config):
    output_dir = hydra.utils.HydraConfig.get().run.dir

    model1_run = mlflow.get_run(config.model1_run_id)
    model2_run = mlflow.get_run(config.model2_run_id)

    model1 = get_pipeline_from_run(model1_run.to_dictionary()['info'])
    model2 = get_pipeline_from_run(model2_run.to_dictionary()['info'])

    dataset1_artifacts = load_dataset_artifacts(model1_run.to_dictionary()['info'])
    dataset2_artifacts = load_dataset_artifacts(model2_run.to_dictionary()['info'])

    feature_dataset1 = FeatureDataset(dataframe=dataset1_artifacts['df'], **dataset1_artifacts['dataset_config'])

    # dataset1 df is used to ensure the same data, different config is used to ensure compatibility
    feature_dataset2 = FeatureDataset(dataframe=dataset1_artifacts['df'], **dataset2_artifacts['dataset_config'])

    f_stat, p_value = combined_ftest_5x2cv(model1, model2, feature_dataset1, feature_dataset2,
                                           save_path=os.path.join(output_dir, 'splits.yml'), multi_class=config.multi_class,
                                           labels=config.labels)

    logger.info(f"f_stat = {f_stat}\np_value = {p_value}")