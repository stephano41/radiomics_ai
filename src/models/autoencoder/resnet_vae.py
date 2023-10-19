"""
Author: Duy-Phuong Dao
Email : phuongdd.1997@gmail.com or duyphuongcri@gmail.com
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_vae import BaseVAE
from .initialisations import he_init


class ResnetVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 finish_size=2) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.finish_size = finish_size

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        num_groups = [1, 8, 16, 16, 16, 16]

        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        for i, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    conv_block(ch_in=in_channels, ch_out=h_dim, k_size=3, num_groups=num_groups[i]),
                    ResNet_block(ch=h_dim, k_size=3, num_groups=num_groups[i + 1]),
                    nn.MaxPool3d(3, stride=2, padding=1)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * finish_size ** 3, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * finish_size ** 3, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Sequential(nn.Linear(latent_dim, hidden_dims[-1] * finish_size ** 3),
                                           nn.ReLU()
                                           )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    up_conv(ch_in=hidden_dims[i], ch_out=hidden_dims[i + 1], k_size=1, scale=2),
                    ResNet_block(ch=hidden_dims[i + 1], k_size=3, num_groups=16)
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            up_conv(ch_in=hidden_dims[-1], ch_out=self.in_channels, k_size=1, scale=2),
            ResNet_block(ch=self.in_channels, k_size=3, num_groups=1)
        )

        self.apply(he_init)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.finish_size, self.finish_size, self.finish_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return (self.decode(z), input, mu, log_var)

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def generate_latent_vars(self, x: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z



class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=1, num_groups=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p, groups=num_groups),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class ResNet_block(nn.Module):
    "A ResNet-like block with the GroupNorm normalization providing optional bottle-neck functionality"

    def __init__(self, ch, k_size, stride=1, p=1, num_groups=1):
        super(ResNet_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p, groups=num_groups),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p, groups=num_groups),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x) + x
        return out


class up_conv(nn.Module):
    "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"

    def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
        )

    def forward(self, x):
        return self.up(x)
