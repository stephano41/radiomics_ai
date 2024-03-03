import logging

from autorad.inference.infer_utils import get_best_run_from_experiment_name, get_last_run_from_experiment_name, \
    load_pipeline_artifacts
from imblearn.pipeline import Pipeline
import mlflow
from autorad.models.classifier import MLClassifier


log = logging.getLogger(__name__)


def get_pipeline_from_last_run(experiment_name="model_training") -> Pipeline:
    run = get_last_run_from_experiment_name(experiment_name)

    return get_pipeline_from_run(run)


def get_pipeline_from_best_run(experiment_name="model_training") -> Pipeline:
    run = get_best_run_from_experiment_name(experiment_name)

    return get_pipeline_from_run(run)


def get_pipeline_from_run(run) -> Pipeline:
    try:
        artifact_uri = run['artifact_uri']
    except IndexError:
        artifact_uri = run.info.artifact_uri
    
    model = MLClassifier.load_from_mlflow(f"{artifact_uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{artifact_uri}/preprocessor")

    pipeline = preprocessor.pipeline
    pipeline.steps.append(['estimator', model])
    return pipeline


def get_pipeline_from_artifacts(artifacts) -> Pipeline:
    pipeline: Pipeline = artifacts["preprocessor"].pipeline
    pipeline.steps.append(['estimator', artifacts['model']])

    return pipeline