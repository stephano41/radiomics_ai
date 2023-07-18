from src import evaluation
from src.evaluation._bootstrap import bootstrap
from src.inference import get_artifacts_from_last_run
from src.pipeline.tune import get_data, get_feature_dataset, wiki_sarcoma_df_merger


dataset = get_data(data_dir='./data',
                   image_stem='image',
                   mask_stem='mask_GTV_Mass')

feature_dataset = get_feature_dataset(dataset,
                                      label_csv_path='./example_data/INFOclinical_STS.csv',
                                      target_column='Grade',
                                      extraction_params='./conf/radiomic_params/mr_default.yaml',
                                      n_jobs=-1,
                                      label_csv_encoding='cp1252',
                                      feature_df_merger={'_target_': 'src.pipeline.wiki_sarcoma_df_merger'})
feature_dataset.split(method='train_with_cross_validation_test')

artifacts = get_artifacts_from_last_run('wiki_sarcoma')

result_df = evaluation.evaluate_feature_dataset(dataset=feature_dataset,
                                                model=artifacts['model'],
                                                preprocessor=artifacts["preprocessor"],
                                                split="test")

print(bootstrap(artifacts['model'], feature_dataset.X, feature_dataset.y, iters=20, num_cpu=1))