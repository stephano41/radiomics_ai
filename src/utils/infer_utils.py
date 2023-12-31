import logging

from autorad.inference.infer_utils import get_artifacts_from_best_run, get_artifacts_from_last_run, \
    load_pipeline_artifacts
from imblearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def get_pipeline_from_last_run(experiment_name="model_training") -> Pipeline:
    artifacts = get_artifacts_from_last_run(experiment_name)

    return get_pipeline_from_artifacts(artifacts)


def get_pipeline_from_best_run(experiment_name="model_training") -> Pipeline:
    artifacts = get_artifacts_from_best_run(experiment_name)

    return get_pipeline_from_artifacts(artifacts)


def get_pipeline_from_run(run) -> Pipeline:
    artifacts = load_pipeline_artifacts(run)
    return get_pipeline_from_artifacts(artifacts)


def get_pipeline_from_artifacts(artifacts) -> Pipeline:
    pipeline: Pipeline = artifacts["preprocessor"].pipeline
    pipeline.steps.append(['estimator', artifacts['model']])

    return pipeline