import torch
import torch.nn as nn
import numpy as np
from lib.util import *
import math


def dis(x, y, xx, yy):
    return math.sqrt((x - xx) ** 2 + (y - yy) ** 2)


class SpatialModel(nn.Module):
    '''
    stations: [(lat1, lon1), (lat2, lon2), ...]
    Input stations list at initialize.
    '''

    def __init__(self, stations):
        super(SpatialModel, self).__init__()
        W = torch.randn(7, requires_grad=True)
        self.W = nn.Parameter(W)
        self.d = [[[0 for _ in range(7)] for _ in range(15)] for _ in range(15)]
        dists = []
        self.stations = []
        for i in range(7):
            self.stations.append(generalID(stations[i][0], stations[i][1]))
        for i in range(15):
            for j in range(15):
                for k in range(7):
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
        for i in range(15):
            for j in range(15):
                for k in range(7):
                    stationx, stationy = self.stations[k] // 15, self.stations[k] % 15
                    res[i, j] += self.proximity(i, j, k) * torch.mul(self.W, x[stationx, stationy])
        return res


if __name__ == '__main__':
    model = SpatialModel(get_air_quality_stations())
    x = np.load("data/air_quality.npy")
    x = torch.from_numpy(x)
    y = model(x[0])
    print(y)
