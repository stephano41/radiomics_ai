import os
os.environ['AUTORAD_RESULT_DIR'] = './outputs'
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src import evaluation
from src.evaluation import bootstrap
from src.inference import get_artifacts_from_last_run
from autorad.config import config
import mlflow

print(f"0. {mlflow.get_tracking_uri()}")
print(os.getenv('AUTORAD_RESULT_DIR'))

artifacts = get_artifacts_from_last_run('wiki_sarcoma')

print(f"1. {mlflow.get_tracking_uri()}")
# result_df = evaluation.evaluate_feature_dataset(dataset=feature_dataset,
#                                                 model=artifacts['model'],
#                                                 preprocessor=artifacts["preprocessor"],
#                                                 split="test")
# model = KNeighborsClassifier(3)
pipeline = artifacts["preprocessor"].pipeline
pipeline.steps.append(['estimator', artifacts['model']])


