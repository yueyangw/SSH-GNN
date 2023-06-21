import torch
import torch.nn as nn
import numpy as np
from lib import *
import math


def dis(x, y, xx, yy):
    return math.sqrt((x - xx) ** 2 + (y - yy) ** 2)


class SpatialViewModel(nn.Module):
    """
    stations: [(lat1, lon1), (lat2, lon2), ...]
    Input stations list at initialize.
    """

    def __init__(self,
                 stations: list,
                 size=(15, 10),
                 qa_features=7,
                 hidden_size=32,
                 nearest_k=3,
                 device='cpu'
                 ):
        super(SpatialViewModel, self).__init__()
        self.size = size
        self.qa_features = qa_features
        self.nearest_k = nearest_k
        self.hidden_size = hidden_size
        self.device = device
        W = torch.randn(qa_features, hidden_size, device=device)
        self.W = nn.Parameter(W)
        self.d = [[[0 for _ in range(len(stations))] for _ in range(size[1])] for _ in range(size[0])]
        dists = []
        self.stations = []
        for i in range(len(stations)):
            self.stations.append(generalID(stations[i][0], stations[i][1]))
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(len(stations)):
                    lon, lat = stations[k][0], stations[k][1]
                    p_id = get_id_by_idx(i, j)
                    p_lat, p_lon = get_latlon_by_id(p_id)
                    dist = dis(p_lat, p_lon, lat, lon)
                    dists.append(dist)
                    self.d[i][j][k] = dist
                self.d[i][j].sort()
        self.delta = np.std(dists)

    def proximity(self, posx, posy, station):
        return math.exp(-(self.d[posx][posy][station] ** 2) / (self.delta ** 2))

    def forward(self, quality_val: torch.Tensor):
        res = torch.zeros(quality_val.size(0), self.size[0], self.size[1], self.hidden_size).to(self.device)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.nearest_k):
                    stationx, stationy = self.stations[k] // self.size[0], self.stations[k] % self.size[1]
                    res[:, i, j] += self.proximity(i, j, k) * torch.matmul(quality_val[:, stationx, stationy], self.W)
        return res


if __name__ == '__main__':
    model = SpatialViewModel(get_air_quality_stations('../../data/point.csv'), (15, 10))
    x = torch.randn(32, 15, 10, 7)
    y = model(x)
    print(y.shape)
