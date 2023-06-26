from model import SSH_GNN_Model
import torch
from torch import nn
from lib import *
import numpy as np
from dataloader import get_train_loader, airdata_scalar


stations = get_air_quality_stations('data/point.csv')
device = 'cuda'


def get_loss(out, y_pre, loss_s, label, qu_val):
    pre_loss = torch.zeros((out.size(0), out.size(1), len(stations), out.size(4)),
                         device=device)
    est_loss = torch.zeros((y_pre.size(1), len(stations), y_pre.size(4)),
                         device=device)
    for i in range(len(stations)):
        stations_id = generalID(stations[i][0], stations[i][1])
        x, y = get_pos_by_id(stations_id)
        est_loss[:, i, :] = (y_pre[-1, :, x, y, :] - qu_val[-1, :, x, y, :]).pow(2)
        pre_loss[:, :, i, :] = (out[:, :, x, y, :] - label[:, :, x, y, :]).pow(2)
    loss_m = pre_loss.mean()
    loss_r = est_loss.mean()
    return loss_m + 0.5 * (loss_r) + 0.1 * loss_s


def train(epochs, model, data_loader, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = 9999999.0
    for epoch in range(epochs):
        loss_arr = []
        for i, (qu_val, mete_val, ctx_val, label) in enumerate(data_loader):
            qu_val = qu_val.permute(1, 0, 2, 3, 4).to(device)
            mete_val = torch.permute(mete_val, (1, 0, 2, 3, 4)).to(device)
            ctx_val = torch.permute(ctx_val, (1, 0, 2, 3, 4)).to(device)
            label = torch.permute(label, (1, 0, 2, 3, 4)).to(device)
            # print(qu_val.shape, mete_val.shape, ctx_val.shape)
            optimizer.zero_grad()

            output, y_pre, loss_s = model(qu_val, mete_val, ctx_val)

            loss = get_loss(output, y_pre, loss_s, label, qu_val)
            loss.backward()

            optimizer.step()
            print("epoch: {}, batch: {}, loss: {}".format(epoch+1, i+1, loss.data))
            loss_arr.append(loss.data.cpu())
            # print(airdata_scalar.inverse_transform(y_pre[0, 0, :, :, 0]))
        if np.mean(loss_arr) < best_loss:
            torch.save(model, "myModel_{}.pth".format(np.mean(loss_arr)))



if __name__ == '__main__':
    train_loader = get_train_loader(batch_size=32, shuffle=True)
    ssh_gnn_model = SSH_GNN_Model(
        qa_size=7,
        mete_size=3,
        ctx_size=12,
        stations=stations,
        map_size=(10, 15),
        func_zone_num=12,
        forecast_time=24,
        forecast_size=7,
        hidden_size=32,
        nearest_k=3,
        activate_alpha=0.2,
        batch_first=False,
        device=device
    ).to(device)

    train(500, ssh_gnn_model, train_loader, 0.001)
