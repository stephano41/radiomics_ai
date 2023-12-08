import torch
from torch import Tensor

from .vanilla_vae import VanillaVAE


class DummyVAE(VanillaVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self = self.to(torch.float)

    def generate_latent_vars(self, x: Tensor, **kwargs) -> Tensor:
        # make sure the device works
        result = torch.tensor([1.0], device=x.device) + torch.tensor([1.0], device=x.device)

        return torch.rand(x.shape[0], self.latent_dim).to(x.device)
