from sklearn.neighbors import KNeighborsClassifier

from src.evaluation import bootstrap
from src.evaluation.roc_curve import plot_roc_curve_with_ci
from src.pipeline.pipeline_components import get_feature_dataset

feature_dataset = get_feature_dataset(target_column='Grade',
                                      existing_feature_df='./outputs/meningioma_feature_dataset.csv')

model = KNeighborsClassifier(5)

scores, tpr_fpr = bootstrap(model, feature_dataset.X, feature_dataset.y, iters=20, num_cpu=1,
                 labels=[0, 1], method='.632')

plot_roc_curve_with_ci(tpr_fpr)
print('done')
