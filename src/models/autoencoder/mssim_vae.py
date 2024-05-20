import torch
from math import exp
from torch import nn, Tensor
from torch.nn import functional as F


class MSSIMLoss(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 window_size: int = 11,
                 size_average: bool = True,
                 kld_weight=0.00025,
                 finish_size=2) -> None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)

        :param in_channels: (Int)
        :param window_size: (Int)
        :param size_average: (Bool)
        """
        super(MSSIMLoss, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average
        self.kld_weight = kld_weight
        self.finish_size = finish_size

    def gaussian_window(self, window_size: int, sigma: float) -> Tensor:
        kernel = torch.tensor([exp((x - window_size // 2) ** 2 / (2 * sigma ** 2))
                               for x in range(window_size)])
        return kernel / kernel.sum()

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _3D_window = _1D_window.view(1, -1, 1, 1) * _1D_window.view(1, 1, -1, 1) * _1D_window.view(1, 1, 1, -1)
        window = _3D_window.expand(in_channels, 1, window_size, window_size, window_size).contiguous()
        return window

    def ssim(self,
             img1: Tensor,
             img2: Tensor,
             window_size: int,
             in_channel: int,
             size_average: bool) -> Tensor:

        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=in_channel)
        mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=in_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=in_channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=in_channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=in_channel) - mu1_mu2

        img_range = 1.0  # img1.max() - img1.min() # Dynamic range
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs

    def forward(self, *args, **kwargs):

        # args from scorch is ([recons, input, mu, log_var] ,X)
        recons = args[0][0]
        input = args[0][1]
        mu = args[0][2]
        log_var = args[0][3]

        # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = self.loss_func(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'recons_loss': recons_loss, 'kld_loss': kld_loss}

    def loss_func(self, img1: Tensor, img2: Tensor) -> Tensor:
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            if img1.shape[-1] <= self.finish_size:
                break
            sim, cs = self.ssim(img1, img2,
                                self.window_size,
                                self.in_channels,
                                self.size_average)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool3d(img1, (2, 2, 2))
            img2 = F.avg_pool3d(img2, (2, 2, 2))

        weights = weights[:len(mssim)]
        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        # if normalize:
        #     mssim = (mssim + 1) / 2
        #     mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output
