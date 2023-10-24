
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.evaluation._bootstrap import Bootstrap
from pytest import mark

@mark.parametrize('num_classes', [2,3])
@mark.parametrize("bootstrap_method", ['oob','.632','.632+'])
def test_bootstrap(tmp_path, bootstrap_method, num_classes):
    # feature_dataset = get_feature_dataset(target_column='Grade',
    #                                       existing_feature_df=os.path.join(os.path.dirname(__file__), 'extracted_features.csv'))
    X = np.random.random([100, 500])
    Y = np.random.randint(0, num_classes, 100)

    model = KNeighborsClassifier(3)
    evaluator = Bootstrap(X, Y, iters=5, num_cpu=1, log_dir=tmp_path, method=bootstrap_method)
    evaluator.run(model)