import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TableAutoEncoderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]