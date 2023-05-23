import torch
import torch.nn as nn
import numpy as np
from lib.util import *
import math


def similarity(xci, xcj):
    similarity = np.exp(-np.linalg.norm(xci - xcj))
    return similarity


class ContextualViewModel(nn.Module):

    def __init__(self, stations: list, size=(15, 10, 6), feature_length=7):
        super(ContextualViewModel, self).__init__()
        self.size = size
        size1 = torch.randn(size)   # context feature
        self.feature_length = feature_length
        W = torch.randn((feature_length, feature_length), dtype=torch.float64, requires_grad=True)
        self.W = nn.Parameter(W)
        self.d = [[[0 for _ in range(feature_length)] for _ in range(size[1])] for _ in range(size[0])]
        self.stations = []
        res = []
        for i in range(feature_length):
            self.stations.append(generalID(stations[i][0], stations[i][1]),)
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(feature_length):
                    xi,xj = stations[k][0], stations[k][1]
                    for f in range(size[2]):
                        ri = size1[xi][xj][f]
                        rj = size1[i][j][f]
                        sim = similarity(ri, rj)
                        res.append(float(sim))
                    self.d[i][j][k] = res

    def forward(self, x: torch.Tensor):
        res = torch.zeros(x.shape)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.feature_length):
                    stationx, stationy = self.stations[k] // self.size[0], self.stations[k] % self.size[1]
                    res[i, j] += self.d[i][j][k] * torch.matmul(x[stationx, stationy], self.W)
        return res


if __name__ == '__main__':

    model = ContextualViewModel(torch.randint(1,9,(7,2)), (15, 10, 6))
    x = np.load("data/air_quality.npy")
    x = torch.from_numpy(x)
    y = model(x[0])
    print(y)
