from monai.networks.nets import SegResNetVAE
import torch
import torch.nn.functional as F


class SegResNetVAE2(SegResNetVAE):
    def forward(self, x):

        net_input = x
        x, _ = self.encode(x)

        vae_input = x

        x_vae = self.vae_down(vae_input)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)
        z_mean = self.vae_fc1(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        if self.vae_estimate_std:
            z_sigma = self.vae_fc2(x_vae)
            z_sigma = F.softplus(z_sigma)
            # vae_reg_loss = 0.5 * torch.mean(z_mean ** 2 + z_sigma ** 2 - torch.log(1e-8 + z_sigma ** 2) - 1)

            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            z_sigma = self.vae_default_std
            # vae_reg_loss = torch.mean(z_mean ** 2)

            x_vae = z_mean + z_sigma * z_mean_rand

        x_vae = self.vae_fc3(x_vae)
        x_vae = self.act_mod(x_vae)
        x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)
        # vae_mse_loss = F.mse_loss(net_input, x_vae)
        # vae_loss = vae_reg_loss + vae_mse_loss
        # return vae_loss
        return x_vae, net_input, z_mean, z_sigma * z_mean_rand

    def generate(self, x):
        return self.forward(x)[0]

    def generate_latent_vars(self, x):
        net_input = x
        x, _ = self.encode(x)

        vae_input = x

        x_vae = self.vae_down(vae_input)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)
        z_mean = self.vae_fc1(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        if self.vae_estimate_std:
            z_sigma = self.vae_fc2(x_vae)
            z_sigma = F.softplus(z_sigma)
            # vae_reg_loss = 0.5 * torch.mean(z_mean ** 2 + z_sigma ** 2 - torch.log(1e-8 + z_sigma ** 2) - 1)

            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            z_sigma = self.vae_default_std
            # vae_reg_loss = torch.mean(z_mean ** 2)

            x_vae = z_mean + z_sigma * z_mean_rand
        return x_vae