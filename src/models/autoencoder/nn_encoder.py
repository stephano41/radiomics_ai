from __future__ import annotations

from importlib import import_module

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from skorch import NeuralNetClassifier
from skorch.callbacks import PassthroughScoring, PrintLog, EpochTimer
from skorch.utils import to_device, to_tensor, to_numpy
import torch
from torch.utils.data import WeightedRandomSampler
from src.models.autoencoder.base_vae import BaseVAE
import os
import random


class NeuralNetEncoder(NeuralNetClassifier, TransformerMixin):
    def __init__(self, module: BaseVAE, output_format='numpy', weighted_sampler=False, weighted_sampler_weights=None, random_seed=123, **kwargs):
        self.random_seed=random_seed
        self.output_format = output_format
        self.weighted_sampler=weighted_sampler
        self.weighted_sampler_weights = weighted_sampler_weights

        set_seed(random_seed)

        super().__init__(**preprocess_kwargs(module=module, **kwargs))

        self.__doc__ = super().__doc__

    def transform(self, X, y=None):
        with torch.no_grad():
            # data_x = to_device((dfsitk2tensor(X) - self._mean) / self._std, self.device)

            dataset = self.get_dataset(X)
            iterator = self.get_iterator(dataset, training=False)
            results = []
            for Xi, _ in iterator:
                results.append(to_numpy(self.module_.generate_latent_vars(to_device(Xi, self.device))))

            results = np.concatenate(results, 0)
            if self.output_format == 'numpy':
                return results
            elif self.output_format == 'pandas':
                return pd.DataFrame(results, columns=[f"{type(self.module_).__name__}_dl_feature_{i}" for i in
                                                      range(results.shape[1])])
            else:
                raise ValueError(f"set_output method not implemented, got {self.output_format}")

    def get_feature_names_out(self):
        pass

    def predict_proba(self, X):
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False):
            if isinstance(yp, tuple):
                yp = yp[4] if len(yp)==5 else yp[0]
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

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

        # allow for returning multiple losses
        if isinstance(loss, dict):
            for name, value in loss.items():
                self.history.record_batch(name, to_numpy(value))
            return loss['loss']

        return loss
    

    def get_weighted_sampler(self, dataset):
        train_targets=torch.tensor(dataset.dataset.y).argmax(axis=1)[dataset.indices]

        if self.weighted_sampler_weights is None:
            class_sample_count = torch.tensor([(train_targets == t).sum() for t in torch.unique(train_targets, sorted=True)])
            weight = 1. / class_sample_count.float()
        else:
            weight = torch.tensor(self.weighted_sampler_weights).float()
        samples_weight = torch.tensor([weight[t] for t in train_targets.type(torch.int)])
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    
    def get_iterator(self, dataset, training=False):
        if training:
            kwargs = self.get_params_for('iterator_train')
            iterator = self.iterator_train

            if self.weighted_sampler:
                kwargs.update({"sampler": self.get_weighted_sampler(dataset), 'shuffle': False})
                
        else:
            kwargs = self.get_params_for('iterator_valid')
            iterator = self.iterator_valid

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size

        if kwargs['batch_size'] == -1:
            kwargs['batch_size'] = len(dataset)

        return iterator(dataset, **kwargs)
    

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            ('print_log', PrintLog()),
            ('valid_kld_loss', PassthroughScoring('kld_loss')),
            ('valid_recons_loss', PassthroughScoring('recons_loss'))
        ]

        # default_callbacks = super()._default_callbacks()
        # return default_callbacks.extend([
        #     ('valid_kld_loss', PassthroughScoring('kld_loss')),
        #     ('valid_recons_loss', PassthroughScoring('recons_loss'))
        # ])


def dfsitk2tensor(df):
    np_df = df.applymap(sitk.GetArrayFromImage)
    result = []
    for row in np_df.iterrows():
        result.append(torch.from_numpy(np.stack(row[1].values)).float())

    return torch.stack(result)


def preprocess_kwargs(**kwargs):
    recognised_kwargs = ['module', 'criterion', 'optimizer', 'iterator_train', 'iterator_valid', 'dataset',
                         'train_split']
    _kwargs = kwargs.copy()

    for k in recognised_kwargs:
        arg = _kwargs.get(k)
        if arg is None:
            continue

        if isinstance(arg, str):
            module = '.'.join(arg.split('.')[:-1])
            name = arg.split('.')[-1]
            _kwargs[k] = getattr(import_module(module, name), name)

    return _kwargs


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False