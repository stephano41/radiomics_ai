import os
from datetime import datetime

import numpy as np
import torch.optim
from sklearn.pipeline import Pipeline
from skorch.helper import SkorchDoctor
from skorch.callbacks import EarlyStopping, GradientNormClipping, Checkpoint

from src.dataset import SitkImageProcessor
from src.evaluation import bootstrap
from src.models import MLClassifier
from src.models.autoencoder import Encoder, VanillaVAE, VAELoss, MSSIM, LogCashLoss, BetaVAELoss
from src.models.autoencoder.nn_encoder import dfsitk2tensor
from src.models.callbacks import SimpleLoadInitState
from src.pipeline.pipeline_components import get_multimodal_feature_dataset, split_feature_dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt

from src.training import Trainer
from src.utils.infer_utils import get_pipeline_from_last_run


def plot_debug(stk_image):
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(stk_image)[5, :, :], cmap='gray')
    plt.show()


def plot_slices(output_tensor, slice_index, num_samples=5, original_tensor=None,
                title=None):
    """
    Plot a slice from each image modality of the output tensor for a specified number of samples.

    Parameters:
        output_tensor (torch.Tensor): The output tensor from the autoencoder.
        slice_index (int): The index of the slice to be plotted.
        image_modalities (list): List of image modality names.
        num_samples (int): The number of samples to plot.
        title_prefix (str): Prefix to add to the plot titles.

    Returns:
        None
    """
    batch_size, num_modalities, length, width, height = output_tensor.shape

    for sample_idx in range(min(num_samples, batch_size)):
        plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

        for modality_idx in range(num_modalities):
            plt.subplot(2, num_modalities, modality_idx + 1)
            plt.imshow(output_tensor[sample_idx, modality_idx, slice_index, :, :], cmap='gray')
            plt.title(f'generated Sample {sample_idx + 1}, {modality_idx}')
            plt.axis('off')

        if original_tensor is not None:
            for modality_idx in range(num_modalities):
                plt.subplot(2, num_modalities, num_modalities+ modality_idx + 1)
                plt.imshow(original_tensor[sample_idx, modality_idx, slice_index, :, :], cmap='gray')
                plt.title(f'original Sample {sample_idx + 1}, {modality_idx}')
                plt.axis('off')

        if title is not None:
            plt.suptitle(title)

        plt.show()

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
sitk_processor = SitkImageProcessor('./outputs', './data/meningioma_data', mask_stem='mask',
                                    image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'), n_jobs=6)


for i, (train_x, train_y, val_x, val_y) in enumerate(zip(feature_dataset.data.X.train_folds, feature_dataset.data.y.train_folds, feature_dataset.data.X.val_folds, feature_dataset.data.y.val_folds)):
    encoder = Encoder(VanillaVAE,
                      module__in_channels=5,
                      module__latent_dim=128,
                      module__hidden_dims=[32, 64, 128],
                      module__finish_size=2,
                      criterion=MSSIM,
                      std_dim=(0, 2, 3, 4),
                      max_epochs=200,
                      output_format='pandas',
                      callbacks=[EarlyStopping(load_best=True),
                                 GradientNormClipping(1),
                                 SimpleLoadInitState(f_optimizer='outputs/saved_models/optimizer.pt',
                                                     f_params='outputs/saved_models/params.pt')],
                      optimizer=torch.optim.AdamW,
                      lr=0.00005,
                      augment_train=True,
                      # criterion__loss_type='B',
                      # criterion__gamma=10.0,

                      # criterion__alpha=10.0,
                      # criterion__beta=1.0
                      criterion__in_channels=5,
                      criterion__window_size=4,
                      transform_kwargs=dict(thetaX=(-90, 90),
                                            thetaY=(-90, 90),
                                            thetaZ=(-90, 90),
                                            tx=(-1, 1),
                                            ty=(-1, 1),
                                            tz=(-1, 1),
                                            scale=(1, 1),
                                            n=10),
                      device='cuda'
                      )

    images = sitk_processor.fit_transform(train_x['ID'])
    encoder.fit(images)

    generated_images = encoder.generate(images)

    destd_images = (generated_images * encoder._std + encoder._mean).numpy()
    plot_slices(destd_images, 8, 2, original_tensor=dfsitk2tensor(images), title=datetime.now().strftime(f"%Y%m%d%H%M%S-fold{i}"))


# encoder.save_params(f_params='outputs/saved_models/params.pt', f_optimizer='outputs/saved_models/optimizer.pt', f_history='outputs/saved_models/history.json')



# print(generated_images)

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

# pipeline = get_pipeline_from_last_run('meningioma')

# idx = np.random.randint(0,115, size=115)
#
# pipeline.fit(feature_dataset.X.loc[idx], feature_dataset.y.loc[idx])

