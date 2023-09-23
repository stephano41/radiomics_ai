import os
from datetime import datetime

import torch.optim
import torchio as tio
from skorch.callbacks import EarlyStopping, GradientNormClipping
from torchio import SubjectsDataset

from src.preprocessing.sitk_image_processor import SitkImageProcessor
from src.dataset.skorch_subject_ds import SkorchSubjectsDataset
from src.dataset.transforming_dataloader import TransformingDataLoader
from src.models.autoencoder import SegResNetVAE2, BetaVAELoss, Encoder
from src.pipeline.pipeline_components import get_multimodal_feature_dataset, split_feature_dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt


def plot_debug(stk_image):
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(stk_image)[5, :, :], cmap='gray')
    plt.show()


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
            plt.savefig(f'{save_dir}_{sample_idx}.png')

        plt.show()


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
                                                 feature_df_merger={
                                                     '_target_': 'src.pipeline.df_mergers.meningioma_df_merger'},
                                                 n_jobs=6,
                                                 existing_feature_df=os.path.join(output_dir, 'extracted_features.csv'),
                                                 additional_features=['ID']
                                                 )

# feature_dataset.df.to_csv("./outputs/meningioma_feature_dataset.csv")

feature_dataset = split_feature_dataset(feature_dataset,
                                        existing_split=os.path.join(output_dir, 'splits.yml'))

sitk_processor = SitkImageProcessor('./data/meningioma_data', mask_stem='mask',
                                    image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'))

subject_list = sitk_processor.subject_list

encoder = Encoder(SegResNetVAE2,
                    module__input_image_size=[64, 64, 64],
                    module__spatial_dims=3,
                    module__in_channels=5,
                    module__out_channels=5,
                    module__dropout_prob=0.2,
                    module__init_filters=32,
                    # output_format='pandas',
                    criterion=BetaVAELoss,
                    max_epochs=200,
                    callbacks=[EarlyStopping(load_best=True),
                               GradientNormClipping(1),

                               # SimpleLoadInitState(f_optimizer='outputs/saved_models/optimizer.pt',
                               #                     f_params='outputs/saved_models/params.pt')
                               ],
                    optimizer=torch.optim.AdamW,
                    batch_size=4,
                    lr=0.001,
                    iterator_train=TransformingDataLoader,
                    iterator_train__augment_transforms=tio.Compose([tio.RandomBlur(std=0.1, label_keys='mask'),
                                                                    tio.RandomAffine(p=0.5, label_keys='mask',
                                                                                     scales=0.1, degrees=0,
                                                                                     translation=0, isotropic=True),
                                                                    tio.RandomFlip(flip_probability=0.5,
                                                                                   label_keys='mask', axes=(0, 1, 2))
                                                                    ]),
                    iterator_train__num_workers=6,
                    iterator_train__shuffle=True,
                    iterator_valid__num_workers=6,
                    dataset=SkorchSubjectsDataset,
                    dataset__transform=tio.Compose([tio.Resample((1, 1, 1)),
                                                    tio.ToCanonical(),
                                                    tio.Mask(masking_method='mask', outside_value=0),
                                                    tio.CropOrPad(target_shape=(64, 64, 64), mask_name='mask'),
                                                    tio.ZNormalization(masking_method='mask')]),
                    # criterion__gamma=10.0,
                    # criterion__max_capacity=1,

                    # criterion__alpha=10.0,
                    # criterion__beta=1.0
                    # criterion__in_channels=5,
                    # criterion__window_size=4,
                    criterion__kld_weight=0.1,
                    # criterion__finish_size=8,
                    device='cuda'
                    )

encoder.fit(subject_list)

generated_images = encoder.predict(subject_list[:10])

subject_dataset = SubjectsDataset(subject_list[:10], transform=tio.Compose([tio.Resample((1, 1, 1)),
                                                    tio.ToCanonical(),
                                                    tio.Mask(masking_method='mask', outside_value=0),
                                                    tio.CropOrPad(target_shape=(64, 64, 64), mask_name='mask'),
                                                    tio.ZNormalization(masking_method='mask')]))

original_images = torch.stack(
    [torch.concatenate([i.data for i in subject.get_images()]) for subject in subject_dataset])
plot_slices(generated_images, original_tensor=original_images, slice_index=32, num_samples=8, title=datetime.now().strftime(f"%Y%m%d%H%M%S"),
            save_dir=f'outputs/generated_images/{datetime.now().strftime(f"%Y%m%d%H%M%S")}')
# for i, (train_x, train_y, val_x, val_y) in enumerate(zip(feature_dataset.data.X.train_folds, feature_dataset.data.y.train_folds, feature_dataset.data.X.val_folds, feature_dataset.data.y.val_folds)):
#     # _encoder = type(encoder)(**encoder.get_params())
#     _encoder = clone(encoder)
#
#     images = sitk_processor.fit_transform(train_x['ID'])
#     _encoder.fit(images)
#
#     generated_images = _encoder.generate(images)
#
#     plot_slices(generated_images, 32, 2, original_tensor=dfsitk2tensor(images), title=datetime.now().strftime(f"%Y%m%d%H%M%S-fold{i}"))

# data_processor = SitkImageProcessor('./outputs', './data/meningioma_data',
#                                     image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
#                                     mask_stem='mask',
#                                     n_jobs=2
#                                     )
print('done')
