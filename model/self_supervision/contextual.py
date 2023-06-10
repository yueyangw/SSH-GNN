import math
import random
import time

import torch
from torch import nn
from lib import *
from model import ApproximateModel


class ContextualInference(nn.Module):

    def __init__(self,
                 approximate_model,
                 stations,
                 map_size=(15, 10),
                 qa_size=7,
                 hidden_size=32,
                 out_size=8,
                 batch_first=False,
                 device='cpu'):
        super(ContextualInference, self).__init__()
        self.approximate_model = approximate_model
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_first = batch_first
        self.map_size = map_size
        self.qa_size = qa_size
        self.device = device
        self.stations = []
        for i in range(len(stations)):
            self.stations.append(generalID(stations[i][0], stations[i][1]))
        self.mse = nn.MSELoss()

    def forward(self, qa_val, mete_val, ctx_val):
        time = qa_val.size(1 if self.batch_first else 0)
        batch = qa_val.size(0 if self.batch_first else 1)

        mask_stations = []
        for x in random.sample(self.stations, math.ceil(len(self.stations) * 0.05)):
            mask_stations.append(x)
        masked_datas = []
        for id in mask_stations:
            x = id // self.map_size[0]
            y = id % self.map_size[1]
            masked_datas.append(qa_val[:, :, x, y, :].clone())
            qa_val[:, :, x, y, :] = torch.zeros_like(qa_val[:, :, x, y, :])

        u_val = self.approximate_model(qa_val, mete_val, ctx_val)
        u_val = torch.reshape(u_val, [time * batch * self.map_size[0] * self.map_size[1], self.hidden_size])
        out = self.linear(u_val)
        out = torch.reshape(out, [time, batch, self.map_size[0], self.map_size[1], self.out_size])

        loss = torch.zeros([]).to(self.device)
        for i in range(len(mask_stations)):
            x = mask_stations[i] // self.map_size[0]
            y = mask_stations[i] % self.map_size[1]
            out_state = out[:, :, x, y, :7]
            label = masked_datas[i]
            loss += self.mse(out_state, label)

        return loss


if __name__ == '__main__':
    device = 'cpu'
    stations = get_air_quality_stations('../../data/point.csv')
    in_model = ApproximateModel((15, 10), stations, 7, 7, 5, 32, 3, False, 0.2, device).to(device)
    batch = 32
    air_quality = torch.randn(12, batch, 15, 10, 7).to(device)
    mete = torch.randn(24, batch, 15, 10, 5).to(device)
    ctx = torch.randn(24, batch, 15, 10, 7).to(device)
    model = ContextualInference(
        in_model,
        stations,
        (15, 10), 7, 32, 8, False, device
    )
    model = model.to(device)
    start = time.time()
    loss = model(air_quality, mete, ctx)
    print(loss)
    loss.backward()
    print(time.time() - start)
