import torch

from src.models.autoencoder import Encoder

import pytest

@pytest.mark.slow
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

