from sklearn.datasets import load_digits
import torch
from skorch.callbacks import PassthroughScoring, PrintLog, EarlyStopping
from src.models.autoencoder import VanillaVAE
from src.models.encoder import Encoder, VAELoss

def test_encoder():


    data = load_digits(n_class=2)

    images, target = data['images'], data['target']

    images = torch.unsqueeze(torch.tensor(images).type(torch.float32), dim=1)

    encoder = Encoder(VanillaVAE,
                      module__in_channels=1,
                      module__latent_dim=100,
                      module__hidden_dims=[32, 64],
                      criterion=VAELoss
                      # callbacks=[
                      #   ('train_loss', PassthroughScoring(
                      #       name='train_loss',
                      #       on_train=True,
                      #   )),
                      #   ('valid_loss', PassthroughScoring(
                      #       name='valid_loss',
                      #   )),
                      #   ('print_log', PrintLog()),
                      #   ('early_stop', EarlyStopping(
                      #       monitor='valid_loss',
                      #       patience=5
                      #   ))]
                      )

    encoder.fit(images)

