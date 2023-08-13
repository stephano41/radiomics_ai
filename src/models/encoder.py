from __future__ import annotations

import numpy as np

from skorch import NeuralNet
from torch.utils.data import Subset


class Encoder(NeuralNet):
    def __init__(self, module, standardise=True, std_dim: int | tuple =3, **kwargs):
        self.standardise=standardise
        self._mean=None
        self._std = None
        self.std_dim = std_dim

        super().__init__(module, **kwargs)

    def transform(self, X, y=None):
        mu, log_var = self.module.encode(X)
        return self.module.reparameterize(mu, log_var)

    def get_split_datasets(self, X, y=None, **fit_params):
        dataset_train, dataset_valid = super().get_split_datasets(X, y, **fit_params)

        if self.standardise:
            train_x = X[dataset_train.indices]
            self._mean = np.mean(train_x, axis=self.std_dim)
            self._std = np.std(train_x, axis=self.std_dim)

            standardised_X = (X - self._mean) / self._std

            standardised_dataset = self.get_dataset(standardised_X, y)

            return Subset(standardised_dataset, dataset_train.indices), Subset(standardised_dataset, dataset_valid.indices)

        return dataset_train, dataset_valid
