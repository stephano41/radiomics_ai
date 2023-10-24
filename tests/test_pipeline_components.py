import numpy as np
import pandas as pd

from pytest import mark
import os

@mark.parametrize("feature_dataset_path", ['./tests/meningioma_feature_dataset.csv', './tests/extracted_features.csv'])
@mark.parametrize("additional_features,autoencoder", [([], None), (['ID'], 'get_dummy_autoencoder')])
def test_auto_preprocessing(tmp_path, feature_dataset_path, additional_features, autoencoder, request):
    os.environ['AUTORAD_RESULT_DIR'] = str(tmp_path)

    from autorad.data import FeatureDataset
    from src.preprocessing import run_auto_preprocessing

    feature_dataset = FeatureDataset(pd.read_csv(feature_dataset_path), ID_colname='ID', target='Grade',
                                     additional_features=additional_features)

    feature_dataset.split(method='train_with_cross_validation')

    if autoencoder is not None:
        autoencoder = request.getfixturevalue(autoencoder)

    run_auto_preprocessing(data=feature_dataset.data,
                           result_dir=tmp_path,
                           use_oversampling=False,
                           use_feature_selection=True,
                           feature_selection_methods=['anova', 'lasso'],
                           autoencoder=autoencoder
                           )


@mark.parametrize("feature_dataset_path", ['./tests/meningioma_feature_dataset.csv', './tests/extracted_features.csv'])
@mark.parametrize("additional_features,autoencoder", [([], None), (['ID'], 'get_dummy_autoencoder')])
def test_trainer(tmp_path, feature_dataset_path, additional_features, autoencoder, request):
    # shutil.copy('./tests/outputs/meningioma_preprocessed.pkl', tmp_path / 'preprocessed.pkl')
    os.environ['AUTORAD_RESULT_DIR'] = str(tmp_path)

    from autorad.data import FeatureDataset
    from autorad.models import MLClassifier
    from src.preprocessing import run_auto_preprocessing
    from src.training import Trainer

    feature_dataset = FeatureDataset(pd.read_csv(feature_dataset_path), ID_colname='ID', target='Grade',
                                     additional_features=additional_features)

    feature_dataset.split(method='train_with_cross_validation')

    if len(np.unique(feature_dataset.y)) >2:
        multi_class_mode = 'ovr'
    else:
        multi_class_mode = 'raise'

    if autoencoder is not None:
        autoencoder = request.getfixturevalue(autoencoder)

    run_auto_preprocessing(data=feature_dataset.data,
                           result_dir=tmp_path,
                           use_oversampling=False,
                           use_feature_selection=True,
                           feature_selection_methods=['anova', 'lasso'],
                           autoencoder=autoencoder
                           )

    models = MLClassifier.initialize_default_sklearn_models()

    trainer = Trainer(
        dataset=feature_dataset,
        models=models,
        result_dir=tmp_path,
        multi_class=multi_class_mode
    )

    trainer.set_optimizer('optuna', n_trials=5)
    trainer.run(auto_preprocess=True, experiment_name='test_model_training')
