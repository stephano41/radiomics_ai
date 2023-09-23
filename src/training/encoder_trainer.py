import json
import os
import itertools
from typing import Tuple

import numpy as np
import pandas as pd
import wandb
from sklearn import clone
from sklearn.model_selection import ParameterGrid
import torch
from skorch import NeuralNet
import matplotlib.pyplot as plt
from skorch.callbacks import WandbLogger

from src.preprocessing import SitkImageProcessor


class EncoderTrainer:
    def __init__(self, autoencoder: NeuralNet, param_grid, feature_dataset, data_dir, image_stems: Tuple[str, ...] = ('image'), mask_stem='mask'):
        self.autoencoder = autoencoder
        self.param_grid = param_grid
        self.feature_dataset = feature_dataset

        self.image_reader = SitkImageProcessor(data_dir, mask_stem=mask_stem,
                                    image_stems=image_stems)

    def run(self, wandb_kwargs, num_samples=5, slice_index=8, save_model=True):
        if wandb_kwargs.get('project', None) is not None:
            existing_runs = load_runs(wandb_kwargs['project'])
        else:
            existing_runs = None

        for params in ParameterGrid(self.param_grid):
            # check if the run has already been done
            if existing_runs is not None:
                for row in existing_runs:
                    if row.items() >= params.items():
                        print(f"already evaluated {params}")
                        pass

            print(f'training {params}')
            _autoencoder2 = clone(self.autoencoder)
            _autoencoder2.set_params(**params)

            all_hyperparameters = _autoencoder2.get_params()
            cleaned_hyperparameters = clean_dict_for_json(all_hyperparameters)

            # Log hyperparameters
            wandb_kwargs.update({'config':cleaned_hyperparameters})

            # Initialize a new run with WandB
            run = wandb.init(**wandb_kwargs)

            _autoencoder2.set_params(callbacks= all_hyperparameters['callbacks'] + [WandbLogger(run, save_model=save_model)])

            for i, (train_x, train_y, val_x, val_y) in enumerate(
                    zip(self.feature_dataset.data.X.train_folds, self.feature_dataset.data.y.train_folds,
                        self.feature_dataset.data.X.val_folds, self.feature_dataset.data.y.val_folds)):

                images = self.image_reader.fit_transform(train_x['ID'])
                _autoencoder = clone(_autoencoder2)
                # Train the autoencoder
                _autoencoder.fit(images)

                generated_images = _autoencoder.predict(images[:num_samples])

                image_ds = _autoencoder.get_dataset(images)

                # Log generated images
                image_plots = plot_generated_images(generated_images, np.stack([image_ds[sample][0].numpy() for sample in range(num_samples)]),
                                                    title=f"{str(params)}-fold-{i}",
                                                    num_samples=num_samples, slice_index=slice_index)

                run.log({f"generated_images_{i}_{k+1}":image_plot for k, image_plot in enumerate(image_plots)})

            # Close the WandB run
            run.finish()


def plot_generated_images(output_image, original_image, num_samples=5, slice_index=8, title=None):
    batch_size, num_modalities, length, width, height = output_image.shape

    images=[]
    for sample_idx in range(min(num_samples, batch_size)):
        plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

        for modality_idx in range(num_modalities):
            plt.subplot(2, num_modalities, modality_idx + 1)
            plt.imshow(output_image[sample_idx, modality_idx, slice_index, :, :], cmap='gray')
            plt.title(f'generated sample {sample_idx + 1}, {modality_idx}')
            plt.axis('off')

        for modality_idx in range(num_modalities):
            plt.subplot(2, num_modalities, num_modalities + modality_idx + 1)
            plt.imshow(original_image[sample_idx, modality_idx, slice_index, :, :], cmap='gray')
            plt.title(f'original sample {sample_idx + 1}, {modality_idx}')
            plt.axis('off')

        if title is not None:
            plt.suptitle(title)

        plt.tight_layout()

        images.append(plt.gcf())

        plt.close('all')

    return images


def clean_dict_for_json(d):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    cleaned_dict = {}
    for key, value in d.items():
        if is_jsonable(value):
            # If the value is already serializable, add it to the cleaned dictionary as-is
            cleaned_dict[key] = value
        elif hasattr(value, "__name__"):
            cleaned_dict[key] = value.__name__
        else:
            # If the value is not serializable and has no '__repr__', replace it with a placeholder
            cleaned_dict[key] = str(type(value))
    return cleaned_dict


def load_runs(project_name):
    api = wandb.Api()
    runs = api.runs(project_name)

    config_list = []
    for run in runs:
        # .config contains the hyperparameters.
        # We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith('_')})
    return config_list