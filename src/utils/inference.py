import logging

import mlflow
import pandas as pd
import os
import pickle
from autorad.inference.infer_utils import get_run_info_as_series

log = logging.getLogger(__name__)


def get_preprocessed_data(run):
    if isinstance(run, str):
        run = get_run_info_as_series(run)

    artifact_uri = run.artifact_uri.removeprefix('file://')

    preprocessed_pkl_path = os.path.join(artifact_uri, 'feature_dataset/preprocessed_data.pkl')
    if not os.path.exists(preprocessed_pkl_path):
        log.warn("no preprocessed data found with associated run, make sure it's in the feature_dataset folder")
        return None
    
    with open(preprocessed_pkl_path, 'rb') as f:
        preprocessed_data = pickle.load(f)

    return preprocessed_data