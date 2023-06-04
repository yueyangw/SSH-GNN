import torch
import torch.nn as nn
import numpy as np
from lib.util import *
import math


class TemporalViewModel(nn.Module):

    """
    input: tensor(hour, batch, x, y, feature)
    output: tensor(batch, x, y, feature)
    """

    def __init__(self,
                 size: tuple,
                 spatial_size: int,
                 meteorology_size: int,
                 context_size: int,
                 batch_first=False,
                 hidden_size=32,
                 device='cpu'
                 ):
        super(TemporalViewModel, self).__init__()
        input_size = spatial_size + meteorology_size + context_size
        self.gru = nn.GRU(input_size, hidden_size=hidden_size, batch_first=batch_first)
        self.size = size
        self.batch_first = batch_first
        self.spatial_size = spatial_size
        self.device = device
        W = torch.randn((hidden_size, spatial_size))
        self.W = nn.Parameter(W)

    def forward(self, spatial, meteorology, context):
        res = torch.zeros((spatial.size(0), self.size[0], self.size[1], self.spatial_size))
        if self.batch_first:
            spatial = spatial.repeat(meteorology.size(1), 1, 1, 1, 1)
        else:
            spatial = spatial.repeat(meteorology.size(0), 1, 1, 1, 1)
        res = res.to(self.device)

        input_x = torch.cat((spatial, meteorology, context), dim=4)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                _, h = self.gru(input_x[:, :, i, j, :])
                res[:, i, j, :] = torch.matmul(h[-1], self.W)

        return res


if __name__ == '__main__':
    model = TemporalViewModel((15, 10), 7, 3, 4)
    spatial = torch.randn((32, 15, 10, 7), dtype=torch.float64)
    meteorology = torch.randn((12, 32, 15, 10, 3), dtype=torch.float64)
    context = torch.randn((12, 32, 15, 10, 4), dtype=torch.float64)
    out = model(spatial, meteorology, context)
    print(out.shape)

