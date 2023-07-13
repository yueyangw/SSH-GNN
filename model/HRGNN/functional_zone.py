import torch.nn as nn
from model.graph_conv import GConv
import torch


class FuncGCN(nn.Module):

    def __init__(self, zone_num, u_features, mete_features, ctx_features, out_features):
        super(FuncGCN, self).__init__()
        self.soft_gcn = GConv(ctx_features, zone_num)
        self.softmax = nn.Softmax(dim=1)
        self.gcn = GConv(u_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.zone_weight = nn.Parameter(torch.rand(mete_features + ctx_features, out_features))

    def forward(self, u_val, mete, ctx, r_adj_mat):
        Srz = self.soft_gcn(ctx, r_adj_mat)
        rz_adj_mat = self.softmax(Srz)

        zone_value = torch.matmul(torch.transpose(rz_adj_mat, 2, 3), u_val)
        z_adj_mat = torch.matmul(torch.matmul(torch.transpose(rz_adj_mat, 2, 3), r_adj_mat), rz_adj_mat)
        zone_output = self.gcn(zone_value, z_adj_mat)
        zone_gate = self.sigmoid(torch.matmul(torch.cat([mete, ctx], dim=3), self.zone_weight))
        region_zone_output = zone_gate * torch.matmul(rz_adj_mat, zone_output)

        return zone_output, region_zone_output


if __name__ == '__main__':
    model = FuncGCN(12, 7, 5, 8, 7)
    u_val = torch.randn(12, 32, 15 * 10, 7)
    mete_val = torch.randn(12, 32, 15 * 10, 5)
    ctx_val = torch.randn(12, 32, 15 * 10, 8)
    adj = torch.randn(15 * 10, 15 * 10)
    y = model(u_val, mete_val, ctx_val, adj)
    print(y[0].shape, y[1].shape)