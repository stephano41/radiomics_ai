from monai.networks.nets import SegResNetVAE


class SegResNetVAE2(SegResNetVAE):
    def forward(self, x):
        net_input = x
        x, down_x = self.encode(x)
        down_x.reverse()

        vae_input = x
        x = self.decode(x, down_x)

        vae_loss = self._get_vae_loss(net_input, vae_input)
        return x, vae_loss
