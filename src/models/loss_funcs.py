import torch.nn as nn


class PassThroughLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args from skorch is ([X, loss], X)
        return args[0][1]