from model import ApproximateModel, HierarchicalGraphModel, ContextualInference, NeighborPredict
from torch import nn
import numpy as np
from lib.util import get_air_quality_stations, generalID
import torch
import math
from lib.util import *


def dis(x, y, xx, yy):
    return math.sqrt((x - xx) ** 2 + (y - yy) ** 2)


class SSH_GNN_Model(nn.Module):

    def __init__(self,
                 qa_size,
                 mete_size,
                 ctx_size,
                 stations,
                 map_size=(15, 10),
                 func_zone_num=12,
                 forecast_time=24,
                 forecast_size=7,
                 hidden_size=32,
                 nearest_k=3,
                 activate_alpha=0.2,
                 batch_first=False,
                 device='cpu'):
        super(SSH_GNN_Model, self).__init__()
        self.stations = stations
        self.approximate = ApproximateModel(
            map_size, stations, ctx_size, qa_size, mete_size,
            hidden_size, nearest_k, batch_first, activate_alpha, device
        )
        self.gnn = HierarchicalGraphModel(
            hidden_size, mete_size, ctx_size, func_zone_num,
            hidden_size, forecast_size, forecast_time, batch_first
        )
        self.contextual = ContextualInference(
            self.approximate, stations, map_size, qa_size,
            hidden_size, forecast_size, batch_first, device
        )
        self.predict_linear = nn.Linear(hidden_size, forecast_size)
        self.neighbor = NeighborPredict(hidden_size)
        self.batch_first = batch_first
        self.device = device
        self.mete_size = mete_size
        self.ctx_size = ctx_size
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.adj_mat = self.get_adj_mat()
        self.forecast_time = forecast_time
        self.forecast_size = forecast_size

    def get_adj_mat(self):
        n = self.map_size[0] * self.map_size[1]
        mat = torch.randn(n, n, device=self.device)
        dists = []
        for i in range(n):
            for j in range(n):
                lat_i, lon_i = get_latlon_by_id(i)
                lat_j, lon_j = get_latlon_by_id(j)
                dist = dis(lat_i, lon_i, lat_j, lon_j)
                mat[i, j] = dist
                dists.append(dist)
        delta = np.std(dists)
        for i in range(n):
            for j in range(n):
                mat[i, j] = math.exp(-(mat[i, j] ** 2) / (delta ** 2))
        return mat

    def forward(self, qa_val, mete_val, ctx_val):
        time = qa_val.size(1 if self.batch_first else 0)
        batch = qa_val.size(0 if self.batch_first else 1)
        x_size = qa_val.size(2)
        y_size = qa_val.size(3)

        u_val = self.approximate(qa_val, mete_val, ctx_val)  # [time, batch, x, y, hidden_size]
        output, h = self.gnn(
            u_val.reshape(time, batch, x_size * y_size, self.hidden_size),
            mete_val[time:].reshape(time, batch, x_size * y_size, self.mete_size),
            ctx_val[time:].reshape(time, batch, x_size * y_size, self.ctx_size),
            self.adj_mat
        )

        _, loss_p = self.neighbor(u_val.reshape(time * batch, x_size * y_size, self.hidden_size), self.adj_mat)
        loss_q = self.contextual(qa_val, mete_val, ctx_val)
        loss_s = loss_q.mean() + loss_p.mean()

        pre_linear_input = u_val.reshape(u_val.size(0) * u_val.size(1) * u_val.size(2) * u_val.size(3), u_val.size(4))
        y_pre = self.predict_linear(pre_linear_input)
        y_pre = y_pre.reshape(u_val.size(0), u_val.size(1), u_val.size(2) * u_val.size(3), y_pre.size(1))

        output = output.reshape(self.forecast_time, batch, self.map_size[0], self.map_size[1], self.forecast_size)
        y_pre = y_pre.reshape(time, batch, self.map_size[0], self.map_size[1], qa_val.size(4))

        return output, y_pre, loss_s

        # pre_linear_input = u_val.reshape(u_val.size(0) * u_val.size(1) * u_val.size(2) * u_val.size(3), u_val.size(4))
        # y_pre = self.predict_linear(pre_linear_input)
        # y_pre = y_pre.reshape(u_val.size(0), u_val.size(1), u_val.size(2) * u_val.size(3), y_pre.size(1))
        # y_loss = torch.zeros((u_val.size(0), u_val.size(1), len(self.stations), u_val.size(4)),
        #                              device=self.device)
        #
        # for i in range(len(self.stations)):
        #     stations_id = generalID(self.stations[i][0], self.stations[i][1])
        #     y_loss[:, :, stations_id, :] = (y_pre[:, :, stations_id, :] - qa_val[:, :, stations_id, :]).pow(2)
        # loss_m = y_loss.sum() / (y_loss.shape.sum())


if __name__ == '__main__':
    model = SSH_GNN_Model(7, 3, 12, (15, 10))
    air_quality = torch.randn(12, 32, 15, 10, 7)
    mete_data = torch.randn(24, 32, 15, 10, 3)
    poi_data = torch.randn(24, 32, 15, 10, 12)
    adj = torch.randn(15 * 10, 15 * 10)
    y = model(air_quality, mete_data, poi_data)
    print(y[0].shape)
