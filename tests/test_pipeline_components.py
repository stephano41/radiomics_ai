import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import pytest
from pytest import mark
import os
from sklearn.datasets import make_classification


def test_get_multimodal_feature_dataset():
    from src.utils.pipeline import get_multimodal_feature_dataset

    print(os.getcwd())

    feature_dataset = get_multimodal_feature_dataset(target_column='Grade',
                                                     data_dir='./data/meningioma_data',
                                                     image_stems=['t2', 'flair'],
                                                     mask_stem='mask',
                                                     label_csv_path='./data/meningioma_meta.csv',
                                                     extraction_params='./conf/radiomic_params/meningioma_mr.yaml',
                                                     n_jobs=-1,
                                                     feature_df_merger={'_target_': 'src.utils.df_mergers.meningioma_df_merger'})

    feature_dataset = get_multimodal_feature_dataset('Grade', existing_feature_df='./tests/meningioma_feature_dataset.csv')


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
                           use_oversampling=True,
                           oversampling_methods=['SMOTE', 'ADASYN', 'BorderlineSMOTE'],
                           use_feature_selection=True,
                           feature_selection_methods=['anova', 'lasso', 'linear_svc', 'tree'],
                           autoencoder=autoencoder
                           )

@mark.parametrize('cfg_tune', [['experiments=meningioma', 'bootstrap.iters=5',
                                'optimizer.n_trials=5']], indirect=True)
def test_model_initialisation(cfg_tune):
    from autorad.models import MLClassifier

    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
    models = [MLClassifier.from_sklearn(model_name) for model_name in OmegaConf.to_container(cfg_tune.models, resolve=True)]

    for model in models:
        # Check if the model initializes without error
        assert model is not None, "Failed to initialize the model."

        # Test model fitting
        try:
            model.fit(X, y)
        except Exception as e:
            pytest.fail(f"Model failed to fit: {e}")

        # Test model prediction
        try:
            predictions = model.predict(X)
            # Verify that predictions are returned and have the correct shape
            assert len(predictions) == len(y), "Predictions and true labels length mismatch."
        except Exception as e:
            pytest.fail(f"Model failed to predict: {e}")


@mark.parametrize("feature_dataset_path", ['./tests/meningioma_feature_dataset.csv', './tests/extracted_features.csv'])
@mark.parametrize("additional_features,autoencoder", [([], None), 
                                                    #   (['ID'], 'get_dummy_autoencoder')
                                                      ])
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
                           use_oversampling=True,
                           oversampling_methods=['SMOTE', 'ADASYN', 'BorderlineSMOTE'],
                           use_feature_selection=True,
                           feature_selection_methods=['anova', 'lasso', 'linear_svc', 'tree'],
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
