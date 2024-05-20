import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.evaluation import Bootstrap
from pytest import mark
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from src.models.autoencoder import NeuralNetEncoder


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

def test_bootstrap_multiprocessing(tmp_path):
    X = np.random.random([100, 500])
    Y = np.random.randint(0, 2, 100)

    encoder = NeuralNetEncoder(module='src.models.autoencoder.DummyVAE',
                               module__in_channels=1,
                               module__latent_dim=64,
                               module__hidden_dims=[8, 16, 32],
                               module__finish_size=2,
                               criterion='src.models.loss_funcs.VAELoss',
                               max_epochs=2,
                               dataset='src.dataset.dummy_dataset.DummyDataset',
                               device='cuda'
                               )
    estimator = KNeighborsClassifier(3)
    model = make_pipeline(encoder, estimator)
    evaluator = Bootstrap(X, Y, iters=12, num_processes=4, log_dir=tmp_path, method='.632')
    evaluator.run(model)


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
