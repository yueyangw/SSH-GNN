import torch
import torch.nn as nn
from model.graph_conv import GConv


class NeighborPredict(nn.Module):

    def __init__(self, num_features):
        super(NeighborPredict, self).__init__()
        self.gcn = GConv(num_features, num_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u_val, r_adj_mat):
        output = self.gcn(u_val, r_adj_mat)
        return output, self.get_loss(u_val, output)

    def get_loss(self, u_val, output):
        u_val = torch.unsqueeze(u_val, dim=2)
        out = torch.einsum('ij,ijk->i', output, u_val)
        out = self.sigmoid(out)
        # TODO: loss函数缺少干扰项
        out = -torch.log(out)
        return out


if __name__ == '__main__':
    model = NeighborPredict(32)
    u = torch.randn(150, 32)
    adj = torch.randn(150, 150)
    loss = model(u, adj)[1].sum()
    loss.backward()
    print(model.gcn.W)
