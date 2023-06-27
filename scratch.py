# import numpy as np
# import six
# from matplotlib import pyplot as plt
#
# from src.dataset import WikiSarcoma
#
# import os
# import pydicom
# from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
# import pytest
# import SimpleITK as sitk
# from radiomics import featureextractor
#
# import os
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import pyplot as plt

# params = os.path.join(os.getcwd(), 'example_settings', 'Params.yaml')
if __name__ == '__main__':
    from autorad.models import MLClassifier
    from autorad.data import FeatureDataset

    feature_dataset = FeatureDataset(pd.read_csv('outputs/extracted_features.csv'),
                                     target="Grade",
                                     ID_colname="ID")

    with open('./outputs/splits.yml', 'r') as f:
        feature_dataset.load_splits((yaml.safe_load(f)))

    models = [
        MLClassifier.from_sklearn("Random Forest", {})
    ]

    # from autorad.preprocessing import run_auto_preprocessing
    #
    # run_auto_preprocessing(
    #     data=feature_dataset.data,
    #     use_feature_selection=True,
    #     feature_selection_methods=["anova", "lasso", "boruta"],
    #     use_oversampling=False,
    #     result_dir=Path('./outputs')
    # )

    from src.training import Trainer

    trainer = Trainer(
        dataset=feature_dataset,
        models=models,
        result_dir='./outputs',
        multi_class='ovr',
        num_classes=3
    )
    #
    experiment_name = 'sarcoma_tutorial'
    #
    trainer.set_optimizer('optuna', n_trials=50)
    trainer.run(auto_preprocess=True, experiment_name=experiment_name)
