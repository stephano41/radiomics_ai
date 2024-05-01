import logging

import mlflow
import pandas as pd
from autorad.inference.infer_utils import get_best_run_from_experiment_name, get_last_run_from_experiment_name
from autorad.models.classifier import MLClassifier
from imblearn.pipeline import Pipeline

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


def get_run_info_as_series(run_id):
    """
    Fetches the MLflow run information for the specified run ID and returns it as a pandas Series.

    Parameters:
        run_id (str): The ID of the run in MLflow.

    Returns:
        pd.Series: A series object containing the run information.

    Raises:
        Exception: If the run ID is not found or an error occurs in fetching the run information.
    """
    try:
        run = mlflow.get_run(run_id)
        run_info = pd.Series(dict(run.info))
        return run_info
    except Exception as e:
        raise Exception(f"Failed to fetch run information for run ID {run_id}: {str(e)}")
