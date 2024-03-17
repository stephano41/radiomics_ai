from collections import defaultdict

import pandas as pd
import numpy as np
from autorad.data import FeatureDataset
import pingouin as pg
from autorad.feature_selection.selector import CoreSelector


class ICCSelector(CoreSelector):
    def __init__(self, icc_type='ICC2', icc_threshold=0.6):
        self.icc_type = icc_type
        self.icc_threshold = icc_threshold
        super().__init__()

    def fit(self, X, y):
        iccs = get_feature_icc(X, y, self.icc_type)
        selected_columns = np.where(iccs['ICC'].to_numpy()>self.icc_threshold)[0]
        self._selected_features = X.columns[selected_columns].to_list()

def get_feature_icc(X: pd.DataFrame, y: pd.Series, icc_type='ICC2'):
    #judges are y
    #wine is feature names
    #score is the value of the feature name

    icc_results = []
    for feature_name in X.columns:
        feature_name_col = generate_feature_name_col(y)
        combined_df = pd.concat([feature_name_col, y, X[feature_name]], axis=1)
        icc_result = pg.intraclass_corr(data=combined_df, targets='Feature Name',raters=y.name, ratings=feature_name, nan_policy='omit')
        icc_result = icc_result.set_index('Type')
        icc_results.append({'Feature Name': feature_name,'ICC':icc_result.loc[icc_type, 'ICC']})
    return pd.DataFrame(icc_results)

def generate_feature_name_col(y):
    seen_counts = defaultdict(int)
    feature_names=[]
    for value in y.to_list():
        feature_names.append(seen_counts[value])
        seen_counts[value] +=1

    return pd.Series(feature_names, name='Feature Name')



from src.models.autoencoder.med3d_resnet import med3d_resnet10,ResNetEncoder
import torch
import time
from src.models.autoencoder import SegResNetVAE2, BetaVAELoss, Encoder
from skorch.callbacks import EarlyStopping, GradientNormClipping
from src.dataset import TransformingDataLoader, SkorchSubjectsDataset
from src.pipeline.pipeline_components import get_multimodal_feature_dataset
import torchio as tio
import matplotlib.pyplot as plt
from datetime import datetime


def plot_slices(output_tensor, slice_index, num_samples=5, original_tensor=None,
                title=None, save_dir=None):
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
                plt.subplot(2, num_modalities, num_modalities + modality_idx + 1)
                plt.imshow(original_tensor[sample_idx, modality_idx, slice_index, :, :], cmap='gray')
                plt.title(f'original Sample {sample_idx + 1}, {modality_idx}')
                plt.axis('off')

        if title is not None:
            plt.suptitle(title)
        if save_dir is not None:
            plt.savefig(save_dir+f'_{sample_idx}.png')

        plt.show()  



feature_dataset = get_multimodal_feature_dataset(data_dir='./data/meningioma_data',
                                                 image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
                                                 mask_stem='mask',
                                                 target_column='Grade',
                                                 label_csv_path='./data/meningioma_meta.csv',
                                                 extraction_params='./conf/radiomic_params/meningioma_mr.yaml',
                                                 feature_df_merger={
                                                     '_target_': 'src.pipeline.df_mergers.meningioma_df_merger'},
                                                 n_jobs=6,
                                                 existing_feature_df='tests/meningioma_feature_dataset.csv',
                                                 additional_features=['ID']
                                                 )
id_list = feature_dataset.X['ID'].to_numpy()

dataset_train_transform = tio.Compose([tio.Resample((1, 1, 1)),
                                                    tio.ToCanonical(),
                                                    tio.Mask(masking_method='mask', outside_value=0),
                                                    tio.CropOrPad(target_shape=(96, 96, 96), mask_name='mask'),
                                                    tio.ZNormalization(masking_method='mask')])

encoder = Encoder(ResNetEncoder,
                    module__input_image_size=[96, 96, 96],
                    module__spatial_dims=3,
                    module__in_channels=5,
                    module__out_channels=5,
                    module__dropout_prob=0.2,
                    module__block='BasicBlock',
                    module__shortcut_type='B',
                    module__blocks_down=[1,1,1,1],
                    module__pretrained_param_path='outputs/pretrained_models/resnet_10_23dataset.pth',
                    batch_size=3,
                    # output_format='pandas',
                    criterion=BetaVAELoss,
                    max_epochs=200,
                    callbacks=[EarlyStopping(load_best=True),
                               GradientNormClipping(1),
                               ],
                    optimizer=torch.optim.AdamW,
                    lr=0.001,
                    iterator_train=TransformingDataLoader,
                    iterator_train__augment_transforms=tio.Compose([tio.RandomGamma(log_gamma=0.1, label_keys=('mask',)),
                                                                    tio.RandomAffine(p=0.5, label_keys=('mask',),
                                                                                     scales=0.1, degrees=0,
                                                                                     translation=0, isotropic=True),
                                                                    tio.RandomFlip(flip_probability=0.5,
                                                                                   label_keys=('mask',), axes=(0, 1, 2))
                                                                    ]),
                    iterator_train__num_workers=15,
                    iterator_train__shuffle=True,
                    iterator_valid__num_workers=4,
                    dataset=SkorchSubjectsDataset,
                    dataset__transform=dataset_train_transform,
                    dataset__data_dir='./data/meningioma_data',
                    dataset__image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
                    dataset__mask_stem='mask',
                    criterion__kld_weight=0.1,
                    device='cuda'
                    )


# restnet10 = med3d_resnet10(input_image_size=[96,96,96], shortcut_type='B', in_channels=5)

encoder.fit(id_list)

with torch.no_grad():
    output, in_x, _, _ = encoder.forward(id_list[:10], training=False, device='cuda')


plot_slices(output.detach().cpu(), 48, 5, in_x.detach().cpu(), "med3d resnet10", f'outputs/encoder_generated_images/med3d_resnet10/{datetime.now().strftime("%Y%m%d-%H%M%S")}')