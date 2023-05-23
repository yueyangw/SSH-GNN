import torch
import torch.nn as nn
import numpy as np
from lib.util import *
import math


def similarity(xci, xcj):
    similarity = np.exp(-np.linalg.norm(xci - xcj))
    return similarity

class ContextualViewModel(nn.Module):

    def __init__(self, size=(15, 15), feature_length=7):
        super(ContextualViewModel, self).__init__()
        self.size = size
        size1 = torch.randn(size)   # context feature
        self.feature_length = feature_length
        W = torch.randn((feature_length, feature_length), dtype=torch.float64, requires_grad=True)
        self.W = nn.Parameter(W)
        self.d = torch.zeros(size)
        for i in range(size[0]-1):
            for j in range(size[1]-1):
                sim=0
                if (i != 0):
                    sim += similarity(size1[i][j], size1[i - 1][j])
                if (i != 0 & j!=0):
                    sim += similarity(size1[i][j], size1[i - 1][j-1])
                if (i != 0 & j != size[1]-1):
                    sim += similarity(size1[i][j], size1[i - 1][j + 1])
                if (j != 0):
                    sim += similarity(size1[i][j], size1[i][j - 1])
                if (j != size[1]-1):
                    sim += similarity(size1[i][j], size1[i][j + 1])
                if (i != size[0]-1):
                    sim += similarity(size1[i][j], size1[i + 1][j])
                if (j != 0 & i!=size[0]-1):
                    sim += similarity(size1[i][j], size1[i +1][j - 1])
                if (j != size[1]-1 & i != size[0]-1):
                    sim += similarity(size1[i][j], size1[i + 1][j + 1])
                self.d[i][j] = sim

    def forward(self, x: torch.Tensor):
        res = torch.zeros(x.shape)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                res[i, j] += self.d[i][j] * torch.matmul(x[i, j], self.W)
        return res


if __name__ == '__main__':

    model = ContextualViewModel((15, 15))
    x = np.load("data/air_quality.npy")
    x = torch.from_numpy(x)
    print(x[0])
    y = model(x[0])
    print(y)
