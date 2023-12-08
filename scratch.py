import os

import shap
from autorad.inference.infer_utils import get_last_run_from_experiment_name, load_dataset_artifacts, \
    load_feature_dataset
from autorad.training.train_utils import log_dataset
from sklearn.neighbors import KNeighborsClassifier

from src.evaluation.stratified_bootstrap import BootstrapGenerator
from src.evaluation._bootstrap import Bootstrap
import numpy as np
import mlflow
import joblib
from sklearn import svm
from src.utils.infer_utils import get_pipeline_from_last_run
from src.pipeline.pipeline_components import get_multimodal_feature_dataset, split_feature_dataset


# Create a Support Vector Machine (SVM) classifier

mlflow.set_tracking_uri("file://"+ os.path.abspath('./outputs/models'))

run = get_last_run_from_experiment_name('meningioma+autoencoder')
pipeline = get_pipeline_from_last_run('meningioma+autoencoder')
dataset_artifacts = load_dataset_artifacts(run)
feature_dataset = load_feature_dataset(feature_df=dataset_artifacts['df'],
                                       dataset_config=dataset_artifacts['dataset_config'],
                                       splits=dataset_artifacts['splits'])

pipeline.fit(feature_dataset.X, feature_dataset.y)