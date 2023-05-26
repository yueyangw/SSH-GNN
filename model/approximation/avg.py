# avg
import torch
import torch.nn as nn
import torch.optim as optim

activation = nn.LeakyReLU()


def aggregate(xdi, xei, xsi):
    return torch.mean(torch.stack([xdi, xei, xsi]), dim=0)


class AirQualityModel(nn.Module):
    def __init__(self):
        super(AirQualityModel, self).__init__()
        self.aggregate = aggregate

    def forward(self, xdi, xei, xsi):
        xui = self.aggregate(xdi, xei, xsi)
        xui = activation(xui)
        return xui


if __name__ == '__main__':
    model = AirQualityModel()
    # loss_fn = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    xdi = torch.randn(15, 10)
    xei = torch.randn(15, 10)
    xsi = torch.randn(15, 10)
    y_true = torch.randn(15, 10)
    # for epoch in range(100):
    #     optimizer.zero_grad()
    #     y_pred = model(xdi, xei, xsi)
    #     loss = loss_fn(y_pred, y_true)
    #     loss.backward()
    #     optimizer.step()

    yri = model(xdi, xei, xsi)
    print(xdi.shape)
    print(yri.shape)
