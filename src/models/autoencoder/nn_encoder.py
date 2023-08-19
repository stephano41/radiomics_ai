from __future__ import annotations

import SimpleITK as sitk
import numpy as np
import torch
from sklearn.base import TransformerMixin

from skorch import NeuralNet
from torch.utils.data import Subset

from src.dataset.dl_dataset import SitkImageTransformer


class Encoder(NeuralNet, TransformerMixin):
    def __init__(self, module, standardise=True, std_dim: int | tuple = 3, augment_train=True, transform_kwargs=None,
                 **kwargs):
        self.standardise = standardise
        self._mean = 0
        self._std = 1
        self.std_dim = std_dim

        self.augment_train = augment_train
        self.image_augmenter = SitkImageTransformer(transform_kwargs)
        super().__init__(module, **kwargs)

    def transform(self, X, y=None):
        mu, log_var = self.module.encode((X - self._mean) / self._std)
        return self.module.reparameterize(mu, log_var)

    def get_split_datasets(self, X, y=None, **fit_params):
        dataset_train, dataset_valid = super().get_split_datasets(np.zeros(len(X)), y, **fit_params)

        # train_data = [X[i] for i in dataset_train.indices]
        # valid_data = [X[i] for i in dataset_valid.indices]
        train_data = X.iloc[dataset_train.indices]
        valid_data = X.iloc[dataset_valid.indices]
        if self.augment_train:
            train_data = self.image_augmenter.transform(train_data)

        train_x = dfsitk2tensor(train_data)
        valid_x = dfsitk2tensor(valid_data)

        if self.standardise:
            # self._mean = np.mean(train_x, axis=self.std_dim)
            # self._std = np.std(train_x, axis=self.std_dim)
            self._mean, self._std = calculate_mean_std(train_x, self.std_dim)

            standardised_train = (train_x - self._mean) / self._std
            standardised_valid = (valid_x - self._mean) / self._std

            return self.get_dataset(standardised_train, standardised_train), self.get_dataset(standardised_valid,
                                                                                              standardised_valid)

        return self.get_dataset(train_x, train_x), self.get_dataset(valid_x, valid_x)

    def evaluation_step(self, batch, training=False):
        return super().evaluation_step((batch - self._mean) / self._std, training)


def dfsitk2tensor(df):
    np_df = df.applymap(sitk.GetArrayFromImage)
    result = []
    for row in np_df.iterrows():
        result.append(torch.from_numpy(np.stack(row[1].values)).float())

    return torch.stack(result)


def calculate_mean_std(X, axis):
    mean = torch.mean(X, dim=axis)
    std = torch.std(X, dim=axis)

    shape = [1 if i in axis else size for i, size in enumerate(X.shape)]

    return mean.reshape(shape), std.reshape(shape)
