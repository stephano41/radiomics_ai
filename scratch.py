import os

from src.pipeline.utils import get_data, get_feature_dataset

from src import evaluation
from src.evaluation import bootstrap
from src.inference import get_artifacts_from_last_run
from autorad.config import config
import mlflow

dataset = get_data(data_dir='./data',
                   image_stem='image',
                   mask_stem='mask_GTV_Mass')

feature_dataset = get_feature_dataset(image_dataset=dataset,
                                      label_csv_path='./example_data/INFOclinical_STS.csv',
                                      target_column='Grade',
                                      extraction_params='./conf/radiomic_params/mr_default.yaml',
                                      n_jobs=-1,
                                      label_csv_encoding='cp1252',
                                      feature_df_merger={'_target_': 'src.pipeline.wiki_sarcoma_df_merger'}
                                      )
