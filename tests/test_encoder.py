from sklearn.datasets import load_digits
import torch
from skorch.callbacks import PassthroughScoring, PrintLog, EarlyStopping
from src.models.autoencoder import VanillaVAE
from src.models.autoencoder import Encoder, VAELoss


def test_encoder():
    data = load_digits(n_class=2)

    images, target = data['images'], data['target']

    images = torch.unsqueeze(torch.tensor(images).type(torch.float32), dim=1)

    encoder = Encoder(VanillaVAE,
                      module__in_channels=1,
                      module__latent_dim=100,
                      module__hidden_dims=[32, 64],
                      criterion=VAELoss
                      )

    encoder.fit(images)

