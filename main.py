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
    # x = np.load('data/air_quality.npy')
    # print(x.shape)
    data = np.load('data/new_data.npz')
    arr1 = data['data']
    print(arr1.shape)

    np.save('data/airdata.npy', arr1)
