import torch
from torch import Tensor

from .vanilla_vae import VanillaVAE


class DummyVAE(VanillaVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self = self.to(torch.float)

    def generate_latent_vars(self, x: Tensor, **kwargs) -> Tensor:
        return torch.rand(x.shape[0], self.latent_dim)
