import os
import pickle
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.dataset import ImageDataset
from src.dataset.dl_dataset import SitkImageProcessor
from src.models import MLClassifier
from src.models.autoencoder import Encoder, VanillaVAE, VAELoss
from src.pipeline.pipeline_components import get_data, get_multimodal_feature_dataset, split_feature_dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import run_auto_preprocessing
from src.training import Trainer
from src.utils.prepro_utils import get_multi_paths_with_separate_folder_per_case


def plot_debug(stk_image):
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(stk_image)[5, :, :], cmap='gray')
    plt.show()

output_dir = './outputs/meningioma/2023-08-20-13-16-02'
# dataset = get_data('./data/meningioma_data', 't1ce', 'mask')
# sitk_images = get_sitk_images(dataset, n_jobs=5)



# setup dataset, extract features, split the data
feature_dataset = get_multimodal_feature_dataset(data_dir='./data/meningioma_data',
                                                 image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
                                                 mask_stem='mask',
                                                 target_column='Grade',
                                                 label_csv_path='./data/meningioma_meta.csv',
                                                 extraction_params='./conf/radiomic_params/meningioma_mr.yaml',
                                                 feature_df_merger={'_target_': 'src.pipeline.df_mergers.meningioma_df_merger'},
                                                 n_jobs=6,
                                                 existing_feature_df= os.path.join(output_dir, 'extracted_features.csv'),
                                                 additional_features=['ID']
                                                 )

# feature_dataset.df.to_csv("./outputs/meningioma_feature_dataset.csv")

feature_dataset = split_feature_dataset(feature_dataset,
                                        existing_split=os.path.join(output_dir,'splits.yml'))

models = MLClassifier.initialize_default_sklearn_models()

# run auto preprocessing
# run_auto_preprocessing(data=feature_dataset.data,
#                        result_dir=Path(output_dir),
#                        use_oversampling= False,
#                        feature_selection_methods=['anova', 'lasso'],
#                        use_feature_selection=True,
#                        autoencoder={'_target_': 'sklearn.pipeline.make_pipeline',
#                                     '_args_':[
#                                         {'_target_': 'src.dataset.dl_dataset.SitkImageProcessor',
#                                          'result_dir': './outputs',
#                                          'data_dir': './data/meningioma_data',
#                                          'image_stems': ('registered_adc', 't2', 'flair', 't1', 't1ce'),
#                                          'mask_stem': 'mask',
#                                          'n_jobs': 6},
#                                         {'_target_': 'src.models.autoencoder.nn_encoder.Encoder',
#                                          'module': 'vanillavae',
#                                          'module__in_channels': 5,
#                                          'module__latent_dim': 256,
#                                          'module__hidden_dims':[16,32, 64],
#                                          'module__finish_size': 2,
#                                          'std_dim':[0, 2, 3, 4],
#                                          'max_epochs': 10,
#                                          'output_format': 'pandas'}]}
# )

trainer = Trainer(
    dataset=feature_dataset,
    models=models,
    result_dir=output_dir,
    multi_class='raise',
    labels=[0,1]
)

trainer.set_optimizer('optuna', n_trials=10)
trainer.run(auto_preprocess=True, experiment_name='meningioma')

# image_paths = get_multi_paths_with_separate_folder_per_case(data_dir='./data/meningioma_data',
#                                                             image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
#                                                             mask_stem='mask',
#                                                             relative=False)

