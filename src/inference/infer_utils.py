import logging

import mlflow
from autorad.inference.infer_utils import get_artifacts_from_best_run
from autorad.utils import mlflow_utils, io
from sklearn.pipeline import Pipeline

from src.models import MLClassifier

log = logging.getLogger(__name__)


def get_last_run_from_experiment_name(experiment_name="model_training"):
    experiment_id = mlflow_utils.get_experiment_id_from_name(experiment_name)
    all_runs = mlflow.search_runs(experiment_ids=experiment_id)
    try:
        last_run = all_runs.iloc[0]
    except IndexError:
        raise IndexError(
            "No trained models found. Please run the training first."
        )

    return last_run


def get_artifacts_from_last_run(experiment_name="model_training"):
    last_run = get_last_run_from_experiment_name(experiment_name)
    artifacts = load_pipeline_artifacts(last_run)

    return artifacts


def get_pipeline_from_last_run(experiment_name="model_training") -> Pipeline:
    artifacts = get_artifacts_from_last_run(experiment_name)

    pipeline: Pipeline = artifacts["preprocessor"].pipeline
    pipeline.steps.append(['estimator', artifacts['model']])

    return pipeline


def get_pipeline_from_best_run(experiment_name="model_training") -> Pipeline:
    artifacts = get_artifacts_from_best_run(experiment_name)

    pipeline: Pipeline = artifacts["preprocessor"].pipeline
    pipeline.steps.append(['estimator', artifacts['model']])

    return pipeline


def load_pipeline_artifacts(run):
    uri = run["artifact_uri"]
    model = MLClassifier.load_from_mlflow(f"{uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{uri}/preprocessor")
    explainer = mlflow.shap.load_explainer(f"{uri}/shap-explainer")
    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "explainer": explainer,
    }
    try:
        extraction_config = io.load_yaml(
            f"{uri.removeprefix('file://')}/feature_extraction/extraction_config.yaml"
        )
        artifacts["extraction_config"] = extraction_config
    except FileNotFoundError:
        log.warning("Feature extraction config not found.")
    return artifacts
