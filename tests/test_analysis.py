import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.shap import plot_dependence_scatter_plot
from src.analysis.hosmer_lemeshow import hosmer_lemeshow
import shap
import sklearn.neural_network
import pandas as pd
from scipy.stats import chi2


@pytest.fixture
def shap_values():
    X, y = shap.datasets.adult()
    model = sklearn.neural_network.MLPClassifier().fit(X, y)
    explainer = shap.Explainer(lambda x: model.predict_log_proba(x)[:, 1], X)
    shap_values = explainer(X[:10])
    return shap_values


def test_plot_dependence_scatter_plot(shap_values, tmpdir):
    save_dir = str(tmpdir)
    n_features = 2
    plot_dependence_scatter_plot(shap_values, n_features, save_dir)

    # Check if the directory is created
    assert os.path.exists(save_dir)

    # Check if the correct number of plots are saved
    assert len(os.listdir(save_dir)) == n_features

    # Check if plots are saved with the correct names
    for idx in range(n_features):
        expected_filename = f"dependence_plot_feature_{idx}.png"
        assert expected_filename in os.listdir(save_dir)


@pytest.fixture
def sample_data():
    np.random.seed(0)
    y_prob = np.random.rand(100)
    y_true = np.random.randint(0, 2, 100)
    return y_prob, y_true


def test_hosmer_lemeshow(sample_data):
    y_prob, y_true = sample_data
    result = hosmer_lemeshow(y_prob, y_true)

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the result DataFrame has the correct shape
    assert result.shape == (1, 2)

    # Check if the p-value is within the range [0, 1]
    assert 0 <= result['p - value'].values[0] <= 1

    # Check if the Chi2 value is non-negative
    assert result['Chi2'].values[0] >= 0
