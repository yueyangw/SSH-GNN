import torch
import torch.nn as nn
from .region import RegionGCN
from .functional_zone import FuncGCN


class HierarchicalGraphModel(nn.Module):

    """
    batch_first = False:
    输入三种特征：
    u_val（多视图近似模型的输出）: size=[time, batch, region_id, hidden_size]
    ctx_val（环境特征）: size=[time, batch, region_id, ctx_size]
    mete_val（气象特征）: size=[time, batch, region_id, mete_size]
    """

    def __init__(self,
                 u_size,
                 mete_size,
                 ctx_size,
                 func_num=12,
                 hidden_size=32,
                 out_size=7,
                 forecast_time=24,
                 batch_first=False):
        super(HierarchicalGraphModel, self).__init__()
        self.gru = nn.GRU(u_size * 2, hidden_size=hidden_size, batch_first=batch_first)
        self.linear1 = nn.Linear(hidden_size, hidden_size * forecast_time)
        self.linear2 = nn.Linear(hidden_size * forecast_time, out_size * forecast_time)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.forecast_time = forecast_time
        self.out_size = out_size
        self.regionGCN = RegionGCN(u_size, mete_size, ctx_size, hidden_size)
        self.funcGCN = FuncGCN(func_num, u_size, mete_size, ctx_size, hidden_size)

    def forward(self, u_val, mete_val, ctx_val, adj_mat):
        if self.batch_first:
            batch_size = u_val.size(0)
        else:
            batch_size = u_val.size(1)
        regions = u_val.size(2)

        region_out = self.regionGCN(u_val, mete_val, ctx_val, adj_mat)
        func_out = self.funcGCN(u_val, mete_val, ctx_val, adj_mat)

        t_features = torch.cat([region_out, func_out[1]], dim=3)
        t_features = t_features.reshape(t_features.size(0), t_features.size(1) * t_features.size(2), t_features.size(3))

        _, h = self.gru(t_features)
        x = torch.squeeze(h)
        x = self.linear1(x)
        x = self.linear2(x)
        output = torch.reshape(x, [batch_size, regions, self.forecast_time, self.out_size])
        output = torch.permute(output, [2, 0, 1, 3])
        return output, h


if __name__ == '__main__':
    model = HierarchicalGraphModel(32, 5, 8, 12, 32, 7, 24, False)
    u_val = torch.randn(12, 32, 15 * 10, 32)
    mete_val = torch.randn(12, 32, 15 * 10, 5)
    ctx_val = torch.randn(12, 32, 15 * 10, 8)
    adj = torch.randn(15 * 10, 15 * 10)
    y = model(u_val, mete_val, ctx_val, adj)
    print(y)


