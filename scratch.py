import os

from autorad.training.train_utils import log_dataset
from sklearn.neighbors import KNeighborsClassifier

from src.evaluation.stratified_bootstrap import BootstrapGenerator
from src.evaluation._bootstrap import Bootstrap
import numpy as np

from src.pipeline.pipeline_components import get_multimodal_feature_dataset, split_feature_dataset

output_dir = './outputs/meningioma/2023-09-01-01-32-46'
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

log_dataset(feature_dataset)