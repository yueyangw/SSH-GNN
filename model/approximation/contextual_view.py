import torch
import torch.nn as nn
import numpy as np
from lib.util import *
import math


def similarity(xci, xcj):
    similarity = np.exp(-np.linalg.norm(xci - xcj))
    return similarity


class ContextualViewModel(nn.Module):

    def __init__(self, stations: list, size=(15, 10), context_length=6):
        super(ContextualViewModel, self).__init__()
        self.size = size
        self.context_length = context_length
        W = torch.randn((context_length, context_length), dtype=torch.float64, requires_grad=True)
        self.W = nn.Parameter(W)
        self.d = [[[0 for _ in range(context_length)] for _ in range(size[1])] for _ in range(size[0])]
        dists = []
        self.stations = []
        for i in range(context_length):
            self.stations.append(generalID(stations[i][0]))
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(context_length):
                    xi = stations[k][0]
                    p_id = get_id_by_idx(i, j)
                    xj = get_latlon_by_id(p_id)
                    sim = similarity(xi, xj)
                    sim.append(sim)
                    self.d[i][j][k] = sim
        self.delta = np.std(dists)

    def forward(self, x: torch.Tensor):
        res = torch.zeros(x.shape)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.context_length):
                    stationx, stationy = self.stations[k] // self.size[0], self.stations[k] % self.size[1]
                    res[i, j] += torch.matmul(x[stationx, stationy], self.W)
        return res


if __name__ == '__main__':
    model = ContextualViewModel(get_air_quality_stations(), (15, 10))
    x = np.load("data/*****")
    x = torch.from_numpy(x)
    y = model(x[0])
    print(y)
