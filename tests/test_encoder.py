import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.models.autoencoder import Encoder
from pytest import mark

def test_nn_encoder():
    # generate random images batch 10, single channel, 16x16x16
    images = torch.randn([10,1,16,16,16]).type(torch.float32)

    encoder = Encoder(module='src.models.autoencoder.VanillaVAE',
                      module__in_channels=1,
                      module__latent_dim=64,
                      module__hidden_dims=[8,16,32],
                      module__finish_size=2,
                      criterion='src.models.autoencoder.VAELoss',
                      max_epochs=2
                      )

    encoder.fit(images)

    encoder.transform(images)

@mark.slow
@mark.parametrize('cfg_tune', [f'experiments={e}' for e in ['meningioma','meningioma_autoencoder','wiki_sarcoma']], indirect=True)
def test_encoder_instantiation(cfg_tune):
    if not cfg_tune.get('preprocessing', {}).get('autoencoder', False):
        return

    HydraConfig().set_config(cfg_tune)
    encoder = instantiate(cfg_tune.preprocessing.autoencoder)

    assert isinstance(encoder, BaseEstimator)

    encoder.fit([f'ID_{i+1}' for i in range(10)])