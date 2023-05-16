import torch
import torch.nn as nn
import numpy as np

class SpatialModel(nn.Module):

    def __init__(self):
        super(SpatialModel, self).__init__()
        W = torch.randn((6, 1), requires_grad=True)
        self.W = nn.Parameter(W)

    def forward(self, x, near_s):
        pass


if __name__ == '__main__':
    model = SpatialModel()
    for k, v in model.named_parameters():
        print(k, v)