import torch.nn as nn
from model.graph_conv import GConv
import torch


class RegionGCN(nn.Module):

    def __init__(self, u_features, mete_features, ctx_features, out_features):
        super(RegionGCN, self).__init__()
        self.gcn = GConv(u_features + mete_features + ctx_features, out_features)

    def forward(self, uxi, mete, ctx, adj_mat):
        input = torch.cat([uxi, mete, ctx], dim=3)
        output = self.gcn(input, adj_mat)
        return output


if __name__ == '__main__':
    model = RegionGCN(7, 5, 8, 7)
    u_val = torch.randn(12, 32, 15 * 10, 7)
    mete_val = torch.randn(12, 32, 15 * 10, 5)
    ctx_val = torch.randn(12, 32, 15 * 10, 8)
    adj = torch.randn(15 * 10, 15 * 10)
    y = model(u_val, mete_val, ctx_val, adj)
    print(y.shape)
