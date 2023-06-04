import torch
import torch.nn as nn
import numpy as np
from lib.util import *
import math


def dist(ri, rj):
    return torch.sqrt((ri[0] - rj[0]) ** 2 + (ri[1] - rj[1]) ** 2)


def similarity(xci, xcj):
    similarity = np.exp(-np.linalg.norm(xci - xcj))
    return similarity


class ContextualViewModel(nn.Module):

    def __init__(self, size=(15, 10), ctx_features=7, hidden_size=32, nearest_k=3, device='cpu'):
        super(ContextualViewModel, self).__init__()
        self.size = size
        size1 = torch.randn(size)  # context feature
        self.ctx_features = ctx_features
        W = torch.randn(ctx_features, hidden_size)
        self.W = nn.Parameter(W)
        self.d = torch.zeros(size)
        self.nearest_k = nearest_k
        self.device = device
        self.hidden_size = hidden_size
        self.nearest_neighbors = torch.zeros((size[0], size[1], nearest_k, 3))
        for i in range(size[0]):
            for j in range(size[1]):
                neighbor_context = []
                for m in range(size[0]):
                    for n in range(size[1]):
                        if m != i or n != j:
                            distance = dist(torch.tensor([i, j]), torch.tensor([m, n]))
                            p_id = get_id_by_idx(m, n)
                            neighbor_context.append((distance, p_id, similarity(size1[i][j], size1[m][n])))
                neighbor_context.sort(key=lambda x: x[0])
                self.nearest_neighbors[i][j] = torch.tensor(neighbor_context[:nearest_k])

    def forward(self, contextual_val: torch.Tensor):
        res = torch.zeros(contextual_val.size(0), self.size[0], self.size[1], self.hidden_size).to(self.device)
        for i in range(self.size[0] - 1):
            for j in range(self.size[1] - 1):
                for self.nearest_k in range(self.nearest_k - 1):
                    stationx, stationy = self.nearest_neighbors[i][j][self.nearest_k][1] // self.size[0], \
                                         self.nearest_neighbors[i][j][self.nearest_k][1] % self.size[1]

                    res[:, i, j] += self.nearest_neighbors[i][j][self.nearest_k][2] * torch.matmul(
                        contextual_val[:, stationx.long().item(), stationy.long().item()], self.W)
        return res


if __name__ == '__main__':
    model = ContextualViewModel((15, 15))
    x = np.load("data/air_quality.npy")
    x = torch.from_numpy(x)
    print(x[0])
    y = model(x[0])
    print(y.shape)
