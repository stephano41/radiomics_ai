import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.evaluation import Bootstrap
from pytest import mark
from sklearn.dummy import DummyClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.datasets import make_classification
from src.evaluation import Bootstrap


# Define a simple model with Skorch
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, X):
        X = X.float()
        X = torch.relu(self.fc1(X))
        X = self.fc2(X)
        return X

@mark.parametrize('num_classes', [2,3])
@mark.parametrize("bootstrap_method", ['oob','.632','.632+'])
def test_bootstrap(tmp_path, bootstrap_method, num_classes):
    # feature_dataset = get_feature_dataset(target_column='Grade',
    #                                       existing_feature_df=os.path.join(os.path.dirname(__file__), 'extracted_features.csv'))
    X = np.random.random([100, 500])
    Y = np.random.randint(0, num_classes, 100)

    model = KNeighborsClassifier(3)
    evaluator = Bootstrap(X, Y, iters=5, num_processes=1, log_dir=tmp_path, method=bootstrap_method)
    evaluator.run(model)


@pytest.mark.gpu
def test_bootstrap_multiprocessing_on_gpu():
    # Generate a dummy dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X = X.astype('float32')

    net = NeuralNetClassifier(
        SimpleNet,
        max_epochs=10,
        lr=0.01,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device='cuda',
        iterator_train__num_workers=3
    )

    # Initialize and run bootstrap
    bootstrap = Bootstrap(X, y, iters=10, num_processes=2, stratify=True, num_gpus=1)
    ci, final_fpr_tpr, final_y_pred = bootstrap.run(net)

    # Add assertions to validate the output
    assert ci is not None, "Confidence intervals should not be None"
    assert isinstance(ci, dict), "Confidence intervals should be a dictionary"
    if final_fpr_tpr:
        assert isinstance(final_fpr_tpr, list), "FPR and TPR should be a list"
    if final_y_pred:
        assert isinstance(final_y_pred, list), "Final predictions should be a list"


def test_combined5x2ftest(tmp_path):
    from autorad.data import FeatureDataset
    from src.evaluation.f_test import combined_ftest_5x2cv

    feature_dataset = FeatureDataset(pd.read_csv('./tests/meningioma_feature_dataset.csv'), ID_colname='ID', target='Grade')

    model1 = DummyClassifier(strategy='most_frequent')
    model2 = DummyClassifier(strategy='uniform')

    f_stat, p_value = combined_ftest_5x2cv(model1, model2, feature_dataset, feature_dataset,
                                           save_path=os.path.join(tmp_path, 'splits.yml'),
                                           multi_class='raise',
                                           labels=[0,1])

    assert f_stat is not None
    assert p_value is not None
