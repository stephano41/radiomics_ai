from __future__ import annotations

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin

from skorch import NeuralNet
from skorch.utils import to_device, to_tensor, to_numpy

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
            # data_x = to_device((dfsitk2tensor(X) - self._mean) / self._std, self.device)

            dataset = self.get_dataset(X)
            iterator = self.get_iterator(dataset, training=False)
            results = []
            for Xi, _ in iterator:

                mu, log_var = self.module_.encode(to_device((Xi - self._mean) / self._std, self.device))
                results.append(to_numpy(self.module_.reparameterize(mu, log_var)))

            results = np.concatenate(results, 0)
            if self.output_format == 'numpy':
                return results
            elif self.output_format == 'pandas':
                return pd.DataFrame(results, columns=[f"{type(self.module_).__name__}_dl_feature_{i}" for i in
                                                     range(results.shape[1])])
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

    def get_feature_names_out(self):
        pass

    def generate(self, x):
        return self.predict_proba(x)

    # def predict_proba(self, X):
    #     """Return the output of the module's forward method as a numpy
    #     array.
    #
    #     If the module's forward method returns multiple outputs as a
    #     tuple, it is assumed that the first output contains the
    #     relevant information and the other values are ignored. If all
    #     values are relevant, consider using
    #     :func:`~skorch.NeuralNet.forward` instead.
    #
    #     Parameters
    #     ----------
    #     X : input data, compatible with skorch.dataset.Dataset
    #       By default, you should be able to pass:
    #
    #         * numpy arrays
    #         * torch tensors
    #         * pandas DataFrame or Series
    #         * scipy sparse CSR matrices
    #         * a dictionary of the former three
    #         * a list/tuple of the former three
    #         * a Dataset
    #
    #       If this doesn't work with your data, you have to pass a
    #       ``Dataset`` that can deal with the data.
    #
    #     Returns
    #     -------
    #     y_proba : numpy ndarray
    #
    #     """
    #     nonlin = self._get_predict_nonlinearity()
    #     y_probas = []
    #     for yp in self.forward_iter(X, training=False):
    #         yp = yp[0] if isinstance(yp, tuple) else yp
    #         yp = nonlin(yp)
    #         y_probas.append(to_numpy(yp))
    #     y_proba = np.concatenate(y_probas, 0)
    #     return y_proba

    def predict_proba(self, X):
        y_proba = super().predict_proba((dfsitk2tensor(X) - self._mean) / self._std)
        return y_proba * to_numpy(self._std) + to_numpy(self._mean)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Return the loss for this batch.

        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values

        y_true : torch tensor
          True target values.

        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        training : bool (default=False)
          Whether train mode should be used or not.

        """
        y_true = to_tensor(y_true, device=self.device)
        loss = self.criterion_(y_pred, y_true)

        if isinstance(loss, dict):
            for name, value in loss.items():
                self.history.record_batch(name, value.data.cpu().detach().numpy())
            return loss['loss']

        return loss


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
