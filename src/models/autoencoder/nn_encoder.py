from __future__ import annotations

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin

from skorch import NeuralNet
from skorch.utils import to_device

from src.dataset import SitkImageTransformer


class Encoder(NeuralNet, TransformerMixin):
    def __init__(self, module, standardise=True, std_dim: int | tuple = 3, transform_kwargs=None,
                 output_format='tensor', image_augmenter=SitkImageTransformer,
                 **kwargs):
        self.standardise = standardise
        self.std_dim = std_dim if isinstance(std_dim, tuple) else tuple(std_dim)
        self.transform_kwargs = transform_kwargs
        self.image_augmenter = image_augmenter
        self.output_format = output_format

        self._mean = 0
        self._std = 1

        super().__init__(**preprocess_kwargs(module=module, **kwargs))

    def transform(self, X, y=None):
        with torch.no_grad():
            data_x = to_device((dfsitk2tensor(X) - self._mean) / self._std, self.device)

            mu, log_var = self.module_.encode(data_x)

            result = self.module_.reparameterize(mu, log_var).detach()
            if self.output_format == 'tensor':
                return result
            elif self.output_format == 'numpy':
                return result.numpy()
            elif self.output_format == 'pandas':
                return pd.DataFrame(result, columns=[f"{type(self.module_).__name__}_dl_feature_{i}" for i in
                                                     range(result.shape[1])])
            else:
                raise ValueError(f"set_output method not implemented, got {self.output_format}")

    def get_split_datasets(self, X, y=None, **fit_params):
        dataset_train, dataset_valid = super().get_split_datasets(np.zeros(len(X)), y, **fit_params)

        # train_data = [X[i] for i in dataset_train.indices]
        # valid_data = [X[i] for i in dataset_valid.indices]
        train_data = X.iloc[dataset_train.indices]
        valid_data = X.iloc[dataset_valid.indices]
        if self.image_augmenter is not None:
            train_data = self.image_augmenter(self.transform_kwargs).transform(train_data)

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

    def get_feature_names_out(self):
        pass

    def generate(self, x):
        with torch.no_grad():
            x = to_device((dfsitk2tensor(x) - self._mean) / self._std, self.device)
            output = to_device(self.module_.generate(x), 'cpu').detach()
            return (output * self._std + self._mean).numpy()


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


def preprocess_kwargs(**kwargs):
    from src.models.autoencoder import VanillaVAE, VAELoss, MSSIM, LogCashLoss, BetaVAELoss

    _kwargs = kwargs.copy()

    module = _kwargs.get('module', None)
    if module is None:
        raise ValueError('module parameter of skorch neural net cannot be empty')
    if isinstance(module, str):
        if module.casefold() == 'vanillavae':
            _kwargs['module'] = VanillaVAE
        else:
            raise ValueError(f"module name not implemented, got {module}")

    optimizer = _kwargs.get('optimizer', torch.optim.SGD)
    if isinstance(optimizer, str):
        if optimizer.casefold() == 'adamw':
            _kwargs['optimizer'] = torch.optim.AdamW
        else:
            raise ValueError(f"optimizer name not implemented, got {optimizer}")

    criterion = _kwargs.get('criterion', None)
    if criterion is None:
        raise ValueError(f"criterion name not implemented, got {criterion}")

    if isinstance(criterion, str):
        if criterion.casefold() == 'mssim':
            _kwargs['criterion'] = MSSIM
        elif criterion.casefold() == 'vaeloss':
            _kwargs['criterion'] = VAELoss
        elif criterion.casefold() == 'logcashloss':
            _kwargs['criterion'] = LogCashLoss
        elif criterion.casefold() == 'betavaeloss':
            _kwargs['criterion'] = BetaVAELoss
        else:
            raise ValueError(f"criterion name not implemented, got {criterion}")

    return _kwargs
