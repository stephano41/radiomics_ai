import os

import numpy as np

from src.evaluation import bootstrap
from src.models import MLClassifier
from src.pipeline.pipeline_components import get_data, get_multimodal_feature_dataset, split_feature_dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt

from src.training import Trainer
from src.utils.infer_utils import get_pipeline_from_last_run


def plot_debug(stk_image):
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(stk_image)[5, :, :], cmap='gray')
    plt.show()

output_dir = './outputs/meningioma/2023-08-24-10-44-04'
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

# models = MLClassifier.initialize_default_sklearn_models()

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
# sitk_processor = SitkImageProcessor('./outputs', './data/meningioma_data', mask_stem='mask',
#                                     image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'), n_jobs=6)
#
# encoder = Encoder(VanillaVAE,
#                   module__in_channels=5,
#                   module__latent_dim=100,
#                   module__hidden_dims= [16, 32, 64],
#                   module__finish_size=2,
#                   criterion=VAELoss,
#                   std_dim=(0,2,3,4),
#                   max_epochs=10,
#                   output_format='pandas',
#                   callbacks=[EarlyStopping(), GradientNormClipping()]
#                   )

# autoencoder_pipeline = Pipeline(steps=[
#     ("read_data", sitk_processor),
#     ('encoder', encoder)
# ])
#
# pipeline = ColumnTransformer(transformers=[('autoencoder', autoencoder_pipeline, 'ID')], remainder='passthrough', verbose_feature_names_out=False)
# pipeline.set_output(transform='pandas')


# trainer = Trainer(
#     dataset=feature_dataset,
#     models=models,
#     result_dir=output_dir,
#     multi_class='raise',
#     labels=[0,1]
# )
#
# trainer.set_optimizer('optuna', n_trials=10)
# trainer.run(auto_preprocess=True, experiment_name='meningioma')

pipeline = get_pipeline_from_last_run('meningioma')

# idx = np.random.randint(0,115, size=115)
#
# pipeline.fit(feature_dataset.X.loc[idx], feature_dataset.y.loc[idx])
if __name__ == '__main__':
    confidence_interval = bootstrap(pipeline, feature_dataset.X, feature_dataset.y,
                                    iters=10,
                                    num_cpu=2,
                                    num_gpu=0,
                                    labels=[0,1],
                                    method='.632+',
                                    stratify=True)

    print(confidence_interval)

