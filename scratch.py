from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src import evaluation
from src.evaluation._bootstrap import bootstrap
from src.inference import get_artifacts_from_last_run
from src.pipeline.tune import get_data, get_feature_dataset, wiki_sarcoma_df_merger


feature_dataset = get_feature_dataset(target_column='Grade',
                                      extraction_params='./conf/radiomic_params/mr_default.yaml',
                                      n_jobs=-1,
                                      label_csv_encoding='cp1252',
                                      feature_df_merger={'_target_': 'src.pipeline.wiki_sarcoma_df_merger'},
                                      existing_feature_df='outputs/wiki_sarcoma/2023-07-16-06-08-02/extracted_features.csv')
feature_dataset.split(method='train_with_cross_validation_test')

# artifacts = get_artifacts_from_last_run('wiki_sarcoma')

# result_df = evaluation.evaluate_feature_dataset(dataset=feature_dataset,
#                                                 model=artifacts['model'],
#                                                 preprocessor=artifacts["preprocessor"],
#                                                 split="test")
model = KNeighborsClassifier(3)

print(bootstrap(model, feature_dataset.X.to_numpy(), feature_dataset.y.to_numpy(), iters=20, num_cpu=1,
                labels=[0,1,2], method='.632+'))