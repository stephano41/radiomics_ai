import mlflow
from autorad.inference.infer_utils import load_pipeline_artifacts
from autorad.utils import mlflow_utils
from sklearn.pipeline import Pipeline


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