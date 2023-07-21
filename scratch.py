import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src import evaluation
from src.evaluation import bootstrap
from src.inference import get_artifacts_from_last_run
from src.pipeline.utils import get_data, get_feature_dataset

print(os.getenv('AUTORAD_RESULT_DIR'))
# dataset = get_data(data_dir='./data',
#                    image_stem='image',
#                    mask_stem='mask_GTV_Mass')

feature_dataset = get_feature_dataset(target_column='Grade',
                                      # image_dataset=dataset,
                                      label_csv_path='./example_data/INFOclinical_STS.csv',
                                      extraction_params='./conf/radiomic_params/mr_default.yaml',
                                      n_jobs=-1,
                                      label_csv_encoding='cp1252',
                                      feature_df_merger={'_target_': 'src.pipeline.wiki_sarcoma_df_merger'},
                                      existing_feature_df='outputs/extracted_features.csv'
                                      )
feature_dataset.split(method='train_with_cross_validation_test')

artifacts = get_artifacts_from_last_run('wiki_sarcoma')

# result_df = evaluation.evaluate_feature_dataset(dataset=feature_dataset,
#                                                 model=artifacts['model'],
#                                                 preprocessor=artifacts["preprocessor"],
#                                                 split="test")
# model = KNeighborsClassifier(3)
pipeline = artifacts["preprocessor"].pipeline
pipeline.steps.append(['estimator', artifacts['model']])

print(bootstrap(pipeline, feature_dataset.X.to_numpy(), feature_dataset.y.to_numpy(), iters=20, num_cpu=1,
                labels=[0, 1, 2], method='.632+'))
