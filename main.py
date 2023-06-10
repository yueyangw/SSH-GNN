import numpy as np
from model.approximation.spatial_view import SpatialViewModel
from lib.util import *
import torch
import torch.nn as nn

if __name__ == '__main__':
    # print(get_air_quality_stations())
    # model = SpatialViewModel(get_air_quality_stations(), (15, 15))
    # x = np.load("data/air_quality.npy")
    # x = torch.from_numpy(x)
    # y = model(x[0])
    # print(y)
    m = nn.Softmax(dim=1)
    input = torch.randn(2, 3)
    output = m(input)
    print(output)
