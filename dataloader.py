import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

indices = []
start1 = 12
end1 = 24
start2 = 0
end2 = 24
class AirQualityDataset(Dataset):
    def __init__(self, file1, file2, file3):
        self.data1 = np.load(file1)
        self.data2 = np.load(file2)
        self.data3 = np.load(file3)


    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        air = self.data1[index + 12:index + 24]
        mete = self.data2[index + 0:index + 24]
        poi = self.data3[:, :, :]
        air_tensor = torch.tensor(air, dtype=torch.float32)
        mete_tensor = torch.tensor(mete, dtype=torch.float32)
        poi_tensor = torch.tensor(poi, dtype=torch.float32)
        poi_tensor = poi_tensor.unsqueeze(0).repeat(24, 1, 1, 1)
        print(poi_tensor.shape)
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