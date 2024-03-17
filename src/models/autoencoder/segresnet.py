import torch
import torch.nn.functional as F
from monai.networks.nets import SegResNetVAE

from .base_vae import BaseVAE


class SegResNetVAE2(SegResNetVAE, BaseVAE):
    def encode(self, x):
        vae_input, _ = super().encode(x)

        x_vae = self.vae_down(vae_input)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)
        z_mean = self.vae_fc1(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        if self.vae_estimate_std:
            z_sigma = self.vae_fc2(x_vae)
            z_sigma = F.softplus(z_sigma)
            # vae_reg_loss = 0.5 * torch.mean(z_mean ** 2 + z_sigma ** 2 - torch.log(1e-8 + z_sigma ** 2) - 1)

            # x_vae = z_mean + z_sigma * z_mean_rand
        else:
            z_sigma = self.vae_default_std
        return z_mean, z_sigma * z_mean_rand

    def forward(self, x):

        net_input = x
        mu, eps = self.encode(x)

        x_vae = self.reparameterize(mu, eps)

        x_vae = self.decode(x_vae)

        # vae_mse_loss = F.mse_loss(net_input, x_vae)
        # vae_loss = vae_reg_loss + vae_mse_loss
        # return vae_loss
        return x_vae, net_input, mu, eps
    
    def decode(self, x_vae):
        x_vae = self.vae_fc3(x_vae)
        x_vae = self.act_mod(x_vae)
        x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)

        return x_vae

    def generate(self, x, **kwargs):
        return self.forward(x)[0]
    
    def reparameterize(self, mu, logvar):
        return mu + logvar
        

    def generate_latent_vars(self, x, **kwargs):
        mu, eps = self.encode(x)
        x_vae = self.reparameterize(mu, eps)
        return x_vae
