import pytest
from pytest import mark

from src.metrics.scorer import Scorer
from sklearn.dummy import DummyClassifier
import numpy as np
from autorad.inference.infer_utils import get_last_run_from_experiment_name
import shap
import mlflow
import joblib
from sklearn import svm

@mark.parametrize('n', [2,3])
def testnclass_scorer(n):
    scorer = Scorer(n>2)

    X = np.random.random([100, 10])
    y = np.random.randint(0, n, size=100)
    clf = DummyClassifier(strategy="most_frequent")
    clf = clf.fit(X, y)
    return scorer(clf, X, y)

@pytest.fixture
def tmp_mlflow_path(tmp_path):
    mlflow_path = tmp_path / "mlflow"
    mlflow_path.mkdir()
    return mlflow_path

def test_svm_mlflow(tmp_mlflow_path):
    svm_classifier = svm.SVC()

    with open("./tests/outputs/preprocessed.pkl", 'rb') as f:
        preprocessed = joblib.load(f)['lasso'][None]
        preprocessed_X = preprocessed.X.train
        preprocessed_Y = preprocessed.y.train

    mlflow.set_tracking_uri(f"file://{tmp_mlflow_path}")
    mlflow.set_experiment('test_experiment')

    svm_classifier.fit(preprocessed_X, preprocessed_Y)

    with mlflow.start_run():
        explainer = shap.Explainer(svm_classifier.predict, preprocessed_X)
        mlflow.shap.log_explainer(explainer, "shap-explainer")

    run = get_last_run_from_experiment_name('test_experiment')

    uri = run["artifact_uri"]
    explainer_uri = f"{uri}/shap-explainer"

    loaded_explainer = mlflow.shap.load_explainer(explainer_uri)

    assert loaded_explainer is not None

    # Cleanup (optional)
    mlflow.end_run()