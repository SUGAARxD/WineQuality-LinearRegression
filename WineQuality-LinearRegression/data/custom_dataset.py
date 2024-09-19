import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

        data_features = self.data.values[:, :-1].astype(np.float32)
        self.mean = np.mean(data_features, axis=0)
        self.std_dev = np.std(data_features, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        features = ((row.iloc[:-1].values.astype(np.float32) - self.mean) / (self.std_dev + 1e-10))
        target = np.float32(row.iloc[-1])

        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32).flatten()

        if self.transform:
            features = self.transform(features)

        return features, target
