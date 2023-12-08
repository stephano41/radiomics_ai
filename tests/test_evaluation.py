
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.evaluation import Bootstrap
from pytest import mark
from sklearn.pipeline import make_pipeline

from src.models.autoencoder import Encoder


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

    encoder = Encoder(module='src.models.autoencoder.DummyVAE',
                      module__in_channels=1,
                      module__latent_dim=64,
                      module__hidden_dims=[8, 16, 32],
                      module__finish_size=2,
                      criterion='src.models.autoencoder.VAELoss',
                      max_epochs=2,
                      dataset='src.dataset.dummy_dataset.DummyDataset',
                      device='cuda'
                      )
    estimator = KNeighborsClassifier(3)
    model = make_pipeline(encoder, estimator)
    evaluator = Bootstrap(X, Y, iters=12, num_processes=4, log_dir=tmp_path, method='.632')
    evaluator.run(model)