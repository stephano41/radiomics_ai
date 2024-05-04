import tempfile
import pickle
import os
import mlflow


def log_preprocessed_data(data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = os.path.join(tmp_dir, 'preprocessed_data.pkl')
        with open(save_dir, 'wb') as f:
            pickle.dump(data, f)
        mlflow.log_artifact(str(save_dir), 'feature_dataset')


def log_hydra(result_dir):
    """
    Checks if a '.hydra' directory exists in the specified result directory.
    If it exists, logs the directory as an artifact in the currently active MLflow run.

    Parameters:
    - result_dir (str): Path to the directory where results (and potentially the .hydra folder) are stored.
    """
    # Construct the path to the .hydra folder
    hydra_path = os.path.join(result_dir, '.hydra')
    
    # Check if the .hydra folder exists
    if os.path.exists(hydra_path) and os.path.isdir(hydra_path):
        # Log the .hydra folder as an artifact
        mlflow.log_artifacts(hydra_path, artifact_path='hydra')
    else:
        # .hydra folder does not exist or is not a directory
        print(f"No .hydra folder found at {hydra_path}.")
