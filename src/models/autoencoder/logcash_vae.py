import torch
from torch import nn


class LogCashLoss(nn.Module):
    def __init__(self, kld_weight=0.00025, alpha: float = 100., beta: float = 10., ):
        super().__init__()
        self.kld_weight = kld_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, *args, **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0][0]
        input = args[0][1]
        mu = args[0][2]
        log_var = args[0][3]
          # Account for the minibatch samples from the dataset
        t = recons - input
        # recons_loss = F.mse_loss(recons, input)
        # cosh = torch.cosh(self.alpha * t)
        # recons_loss = (1./self.alpha * torch.log(cosh)).mean()

        recons_loss = self.alpha * t + \
                      torch.log(1. + torch.exp(- 2 * self.alpha * t)) - \
                      torch.log(torch.tensor(2.0))
        # print(self.alpha* t.max(), self.alpha*t.min())
        recons_loss = (1. / self.alpha) * recons_loss.mean()

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.beta * self.kld_weight * kld_loss
        return loss
