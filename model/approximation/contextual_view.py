import torch
import torch.nn as nn
import numpy as np
from lib.util import *
import math
K=8
def dist(ri, rj):
    return torch.sqrt((ri[0] - rj[0]) ** 2 + (ri[1] - rj[1]) ** 2)

def similarity(xci, xcj):
    similarity = np.exp(-np.linalg.norm(xci - xcj))
    return similarity

class ContextualViewModel(nn.Module):

    def __init__(self, size=(15, 15), feature_length=7):
        super(ContextualViewModel, self).__init__()
        self.size = size
        size1 = torch.randn(size)   # context feature
        self.feature_length = feature_length
        W = torch.randn((feature_length, feature_length), dtype=torch.float64, requires_grad=True)
        self.W = nn.Parameter(W)
        self.d = torch.zeros(size)
        self.nearest_neighbors = torch.zeros((size[0],size[1],K,3))
        for i in range(size[0]):
            for j in range(size[1]):
                neighbor_context = []
                for m in range(size[0]):
                    for n in range(size[1]):
                        if m != i or n != j:
                            distance = dist(torch.tensor([i,j]), torch.tensor([m, n]))
                            p_id = get_id_by_idx(m, n)
                            neighbor_context.append((distance,p_id, similarity(size1[i][j],size1[m][n])))
                neighbor_context.sort(key=lambda x: x[0])
                self.nearest_neighbors[i][j] = torch.tensor(neighbor_context[:K])

    def forward(self, x: torch.Tensor):
        res = torch.zeros(x.shape)
        for i in range(self.size[0]-1):
            for j in range(self.size[1]-1):
                for k in range(K-1):
                    stationx, stationy = self.nearest_neighbors[i][j][k][1] // self.size[0], self.nearest_neighbors[i][j][k][1] % self.size[1]

                    res[i, j] += self.nearest_neighbors[i][j][k][2] * torch.matmul(x[stationx.long().item(), stationy.long().item()], self.W)
        return res


if __name__ == '__main__':

    model = ContextualViewModel((15, 15))
    x = np.load("data/air_quality.npy")
    x = torch.from_numpy(x)
    print(x[0])
    y = model(x[0])
    print(y.shape)
