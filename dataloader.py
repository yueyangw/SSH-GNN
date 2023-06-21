import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

indices = []
start1 = 12
end1 = 24
start2 = 0
end2 = 24


class AirQualityDataset(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __len__(self):
        return self.data1.size(0) - 47

    def __getitem__(self, index):
        air = self.data1[index + 12:index + 24]
        mete = self.data2[index + 0:index + 24]
        poi = self.data3
        poi_tensor = poi.unsqueeze(0).repeat(24, 1, 1, 1)
        label = self.data1[index + 24: index + 48]
        return air, mete, poi_tensor, label


class ZScoreScalar:
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def transform(self, data):
        return (data - self._mean) / self._std

    def inverse_transform(self, data):
        return data * self._std + self._mean


airdata = torch.Tensor(np.load("data/airdata.npy"))
metedata = torch.Tensor(np.load("data/metedata.npy"))
poidata = torch.Tensor(np.load("data/poidata.npy"))

airdata_scalar = ZScoreScalar(airdata.mean(), airdata.std())
metedata_scalar = ZScoreScalar(metedata.mean(), metedata.std())
poidata_scalar = ZScoreScalar(poidata.mean(), poidata.std())

dataset = AirQualityDataset(
    airdata_scalar.transform(airdata),
    metedata_scalar.transform(metedata),
    poidata_scalar.transform(poidata),
)
# print("---", dataset[0][3][0, :, :, 2])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

def get_train_loader(batch_size=32, shuffle=True):
    return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)


def get_test_loader(batch_size=32, shuffle=True):
    return DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)


# for batch_features, batch_targets in dataloader:

if __name__ == '__main__':
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    for x in test_loader:
        print(x[1].shape)
