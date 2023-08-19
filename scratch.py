import os
import pickle

import pandas as pd
import torch
import yaml

from src.dataset import ImageDataset
from src.pipeline.pipeline_components import get_data, get_multimodal_feature_dataset, split_feature_dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt
from src.dataset.dl_dataset import SitkImageProcessor
import numpy as np
from src.models.autoencoder import VanillaVAE, VAELoss, Encoder
from src.utils.prepro_utils import get_multi_paths_with_separate_folder_per_case


def plot_debug(stk_image):
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(stk_image)[5, :, :], cmap='gray')
    plt.show()


# dataset = get_data('./data/meningioma_data', 't1ce', 'mask')
# sitk_images = get_sitk_images(dataset, n_jobs=5)



# setup dataset, extract features, split the data
# feature_dataset = get_multimodal_feature_dataset(data_dir='./data/meningioma_data',
#                                                  image_stems=('t2', 'flair'),
#                                                  mask_stem='mask',
#                                                  target_column='Grade',
#                                                  label_csv_path='./data/meningioma_meta.csv',
#                                                  extraction_params='./conf/radiomic_params/meningioma_mr.yaml',
#                                                  feature_df_merger={'_target_': 'src.pipeline.tune.meningioma_df_merger'},
#                                                  existing_feature_df= './outputs/meningioma_feature_dataset.csv')
#
# feature_dataset = split_feature_dataset(feature_dataset,
#                                         method='train_with_cross_validation',
#                                         n_splits=5)
paths_df = get_multi_paths_with_separate_folder_per_case('./data/meningioma_data',
                                                         relative=False,
                                                         image_stems=('t2', 'flair', 'registered_adc', 't1', 't1ce'),
                                                         mask_stem='mask',
                                                         )

sitk_processor = SitkImageProcessor('./outputs', paths_df, mask_stem='segmentation_path', image_column_prefix='image_', n_jobs=6)

X = sitk_processor.transform(paths_df.ID)

#
from skorch.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline


encoder = Encoder(VanillaVAE,
                  module__in_channels=5,
                  module__latent_dim=100,
                  module__hidden_dims= [16, 32, 64],
                  module__finish_size=2,
                  criterion=VAELoss,
                  std_dim=(0,2,3,4),
                  max_epochs=10,
                  # dataset=SitkDataset
                  # callbacks=[
                  #   ('early_stop', EarlyStopping(
                  #       monitor='valid_loss',
                  #       patience=5
                  #   ))]
                  )

encoder.fit(X)