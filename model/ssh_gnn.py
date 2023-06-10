from approximation import ApproximateModel
from HRGNN import HierarchicalGraphModel
from self_supervision import ContextualInference, NeighborPredict
from torch import nn
import torch
from lib.util import *


class SSH_GNN_Model(nn.Module):

    def __init__(self,
                 qa_size,
                 mete_size,
                 ctx_size,
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
        stations = get_air_quality_stations('../../data/point.csv')
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
        self.neighbor = NeighborPredict(hidden_size)
        self.batch_first = batch_first

    def forward(self, qa_val, mete_val, ctx_val, adj_mat):
        time = qa_val.size(1 if self.batch_first else 0)
        batch = qa_val.size(0 if self.batch_first else 1)

        u_val = self.approximate(qa_val, mete_val, ctx_val)  # [time, batch, x, y, hidden_size]
        output, h = self.gnn(u_val, mete_val, ctx_val, adj_mat)

        loss_p = self.neighbor(u_val, adj_mat)
        loss_q = self.contextual(qa_val, mete_val, ctx_val)
        loss_s = loss_q.sum() + loss_p.sum()



