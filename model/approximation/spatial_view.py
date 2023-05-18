import torch
import torch.nn as nn
import numpy as np
from lib.util import *
import math


def dis(x, y, xx, yy):
    return math.sqrt((x - xx) ** 2 + (y - yy) ** 2)


class SpatialViewModel(nn.Module):
    """
    stations: [(lat1, lon1), (lat2, lon2), ...]
    Input stations list at initialize.
    """

    def __init__(self, stations: list, size=(15, 10), feature_length=7):
        super(SpatialViewModel, self).__init__()
        self.size = size
        self.feature_length = feature_length
        W = torch.randn((feature_length, feature_length), dtype=torch.float64, requires_grad=True)
        self.W = nn.Parameter(W)
        self.d = [[[0 for _ in range(feature_length)] for _ in range(size[1])] for _ in range(size[0])]
        dists = []
        self.stations = []
        for i in range(feature_length):
            self.stations.append(generalID(stations[i][0], stations[i][1]))
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(feature_length):
                    lon, lat = stations[k][0], stations[k][1]
                    p_id = get_id_by_idx(i, j)
                    p_lat, p_lon = get_latlon_by_id(p_id)
                    dist = dis(p_lat, p_lon, lat, lon)
                    dists.append(dist)
                    self.d[i][j][k] = dist
        self.delta = np.std(dists)

    def proximity(self, posx, posy, station):
        return math.exp(-self.d[posx][posy][station] / (self.delta ** 2))

    def forward(self, x: torch.Tensor):
        res = torch.zeros(x.shape)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.feature_length):
                    stationx, stationy = self.stations[k] // self.size[0], self.stations[k] % self.size[1]
                    res[i, j] += self.proximity(i, j, k) * torch.matmul(x[stationx, stationy], self.W)
        return res


if __name__ == '__main__':
    model = SpatialViewModel(get_air_quality_stations(), (15, 15))
    x = np.load("data/air_quality.npy")
    x = torch.from_numpy(x)
    y = model(x[0])
    print(y)
