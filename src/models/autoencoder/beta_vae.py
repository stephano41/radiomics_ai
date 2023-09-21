import torch
from torch import nn
from torch.nn import functional as F


class BetaVAELoss(nn.Module):
    num_iter = 0
    def __init__(self, beta=4, gamma=1000, kld_weight=0.00025, max_capacity=25, Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',):
        super().__init__()
        self.beta=beta
        self.gamma=gamma
        self.kld_weight = kld_weight
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

    def forward(self, *args, **kwargs):
        self.num_iter += 1
        recons = args[0][0]
        input = args[0][1]
        mu = args[0][2]
        log_var = args[0][3]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * self.kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * self.kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'recons_loss': recons_loss, 'kld_loss': kld_loss}
