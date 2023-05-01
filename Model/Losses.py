import torch
from torch import nn


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target):
        return torch.mean((torch.sum(torch.pow(input - target, 2), dim=1)))
