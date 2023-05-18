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
                 batch_size: int,
                 spatial_size: int,
                 meteorology_size: int,
                 context_size: int,
                 size: tuple,
                 batch_first=False,
                 hidden_size=32,
                 ):
        super(TemporalViewModel, self).__init__()
        input_size = spatial_size + meteorology_size + context_size
        self.gru = nn.GRU(input_size, hidden_size=hidden_size, batch_first=batch_first, dtype=torch.float64)
        self.h = torch.zeros(batch_size, hidden_size)
        self.size = size
        self.batch_size = batch_size
        self.spatial_size = spatial_size
        W = torch.randn((hidden_size, spatial_size), dtype=torch.float64)
        self.W = nn.Parameter(W)

    def forward(self, spatial, meteorology, context):
        input_x = torch.cat((spatial, meteorology, context), dim=4)
        res = torch.zeros((self.batch_size, self.size[0], self.size[1], self.spatial_size))

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                _, h = self.gru(input_x[:, :, i, j, :])
                res[:, i, j, :] = torch.matmul(h[0], self.W)

        return res


if __name__ == '__main__':
    model = TemporalViewModel(32, 7, 3, 4, (15, 10))
    spatial = torch.randn((12, 32, 15, 10, 7), dtype=torch.float64)
    meteorology = torch.randn((12, 32, 15, 10, 3), dtype=torch.float64)
    context = torch.randn((12, 32, 15, 10, 4), dtype=torch.float64)
    out = model(spatial, meteorology, context)
    print(out.shape)

