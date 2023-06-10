import math
import torch
import torch.nn as nn


class GConv(nn.Module):

    def __init__(self, in_size, out_size, bias=True):
        super(GConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.W = nn.Parameter(torch.FloatTensor(in_size, out_size))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_size))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()
        self.relu = nn.LeakyReLU(0.2)

    def init_parameters(self):
        std = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input, adjacency_mat):
        support = torch.matmul(input, self.W)
        output = torch.matmul(adjacency_mat, support)
        if self.bias is not None:
            return self.relu(output + self.bias)
        else:
            return self.relu(output)
