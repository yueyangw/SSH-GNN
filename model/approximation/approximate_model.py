# avg
import time

import torch
import torch.nn as nn
from lib import get_air_quality_stations
from model.approximation.contextual_view import ContextualViewModel
from model.approximation.temporal_view import TemporalViewModel
from model.approximation.spatial_view import SpatialViewModel


def aggregate(xdi, xei, xsi):
    return torch.mean(torch.stack([xdi, xei, xsi]), dim=0)


class ApproximateModel(nn.Module):

    """
    batch_first=False:
    输入三种特征数据
    qa_val（空气质量）size=[time(-24~0), batch, x, y, features(7)],
    mete_val（天气预报）size=[time(-24~0), batch, x, y, features(7)],
    ctx_val（环境数据）size=[time(-24~0), batch, x, y, features(7)],
    """
    def __init__(self,
                 map_size,
                 stations,
                 ctx_features,
                 qa_features,
                 mete_features,
                 hidden_size=32,
                 nearest_k=3,
                 batch_first=False,
                 activate_alpha=0.2,
                 device='cpu'):
        super(ApproximateModel, self).__init__()
        self.contextual_model = ContextualViewModel(map_size, ctx_features, hidden_size, nearest_k, device)
        self.spatial_model = SpatialViewModel(stations, map_size, qa_features, hidden_size, nearest_k, device)
        self.temporal_model = TemporalViewModel(map_size, hidden_size, mete_features, ctx_features, batch_first,
                                                hidden_size, device)
        self.activation = nn.LeakyReLU(activate_alpha)
        self.device = device
        self.batch_first = batch_first
        self.map_size = map_size
        self.hidden_size = hidden_size
        self.qa_size = qa_features

    def forward(self, qa_val, mete_val, ctx_val):
        batch_pos = 0 if self.batch_first else 1
        time_pos = 1 - batch_pos
        time = qa_val.size(time_pos)
        batch = qa_val.size(batch_pos)
        ans = torch.zeros(time, batch, self.map_size[0], self.map_size[1], self.hidden_size).to(self.device)
        qa_val = torch.reshape(qa_val, [time * batch, self.map_size[0], self.map_size[1], self.qa_size])
        spatial_out = self.spatial_model(qa_val)
        spatial_out = torch.reshape(spatial_out, [time, batch, self.map_size[0], self.map_size[1], self.hidden_size])
        for t in range(time):
            temporal_out = self.temporal_model(spatial_out[t], mete_val[t:t+time], ctx_val[t:t+time])
            contextual_out = self.contextual_model(ctx_val[t+time])
            output = aggregate(spatial_out[t], temporal_out, contextual_out)
            output = self.activation(output)
            ans[t] = output
        return ans


if __name__ == '__main__':
    device = 'cpu'
    stations = get_air_quality_stations('../../data/point.csv')
    model = ApproximateModel((15, 10), stations, 7, 7, 5, 32, 3, False, 0.2, device).to(device)
    batch = 32
    air_quality = torch.randn(12, batch, 15, 10, 7).to(device)
    mete = torch.randn(24, batch, 15, 10, 5).to(device)
    ctx = torch.randn(24, batch, 15, 10, 7).to(device)
    s = time.time()
    y = model(air_quality, mete, ctx)
    print(time.time() - s, y.shape)
