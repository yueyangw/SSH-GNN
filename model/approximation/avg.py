# avg
import torch
import torch.nn as nn
import torch.optim as optim

input_size = 15 * 10
output_size = 15 * 10
activation = nn.LeakyReLU()
def aggregate(xdi, xei, xsi):
    return torch.mean(torch.stack([xdi, xei, xsi]), dim=0)

class AirQualityModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AirQualityModel, self).__init__()
        self.aggregate = aggregate
        self.weight = nn.Parameter(torch.randn(input_size, input_size))  # 学习的矩阵

    def forward(self, xdi, xei, xsi):
        xui = self.aggregate(xdi, xei, xsi)
        xui = torch.matmul(xui, self.weight)  # 加权矩阵乘法
        xui = activation(xui)
        return xui


if __name__ == '__main__':
    model = AirQualityModel(input_size, output_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    xdi = torch.randn(1, input_size)
    xei = torch.randn(1, input_size)
    xsi = torch.randn(1, input_size)
    y_true = torch.randn(1, output_size)
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(xdi, xei, xsi)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

    yri = model(xdi, xei, xsi)
    print(xdi.shape)
    print(yri.shape)