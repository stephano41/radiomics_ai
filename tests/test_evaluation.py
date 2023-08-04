import os.path

from sklearn.neighbors import KNeighborsClassifier

from src.evaluation import bootstrap
from src.pipeline.pipeline_components import get_feature_dataset


def test_bootstrap_632plus():
    _test_bootstrap('.632+')


def test_bootstrap_632():
    _test_bootstrap('.632')


def test_bootstrap_oob():
    _test_bootstrap('oob')


def _test_bootstrap(bootstrap_method):
    feature_dataset = get_feature_dataset(target_column='Grade',
                                          existing_feature_df=os.path.join(os.path.dirname(__file__), 'extracted_features.csv'))

    model = KNeighborsClassifier(3)

    return bootstrap(model, feature_dataset.X.to_numpy(), feature_dataset.y.to_numpy(), iters=20, num_cpu=1,
                     labels=[0, 1, 2], method=bootstrap_method)