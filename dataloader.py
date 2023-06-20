import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AirQualityDataset(Dataset):
    def __init__(self, file1, file2, file3):
        self.data1 = np.load(file1)
        self.data2 = np.load(file2)
        self.data3 = np.load(file3)

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        air = self.data1[index]
        mete = self.data2[index]
        poi = self.data3[index]
        air_tensor = torch.tensor(air, dtype=torch.float32).unsqueeze(dim=0)
        mete_tensor = torch.tensor(mete, dtype=torch.float32).unsqueeze(dim=0)
        poi_tensor = torch.tensor(poi, dtype=torch.float32)
        poi_tensor = poi_tensor.unsqueeze(0).repeat(695, 1, 1, 1)
        poi_tensor = poi_tensor.unsqueeze(0)
        return air_tensor, mete_tensor, poi_tensor


file1 = "data/airdata.npy"
file2 = "data/metedata.npy"
file3 = "data/poidata.npy"
batch_size = 32
shuffle = True

dataset = AirQualityDataset(file1, file2, file3)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
for idx, train in enumerate(dataloader):
    print(idx, train)
    break

# for batch_features, batch_targets in dataloader: