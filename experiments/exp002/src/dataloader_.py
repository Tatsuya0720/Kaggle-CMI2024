import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

class LSTMAutoEncoderDataset(Dataset):
    def __init__(self, valid_ids, base_dir):
        self.base_dir = base_dir
        self.valid_ids = valid_ids
        self.scale_columns = [
            "X",
            "Y",
            "Z",
            "enmo",
            "anglez",
            "light",
            "battery_voltage",
        ]

        self.masked_columns = [
            "masked_X",
            "masked_Y",
            "masked_Z",
            "masked_enmo",
            "masked_anglez",
            "masked_light",
        ]

        self.original_columns = [
            'battery_voltage',
            'non-wear_flag'
        ]
        

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        p = read_parquet(self.base_dir, self.valid_ids[idx])
        p["masked_X"] = p["X"] * p["non-wear_flag"]
        p["masked_Y"] = p["Y"] * p["non-wear_flag"]
        p["masked_Z"] = p["Z"] * p["non-wear_flag"]
        p["masked_enmo"] = p["enmo"] * p["non-wear_flag"]
        p["masked_anglez"] = p["anglez"] * p["non-wear_flag"]
        p["masked_light"] = p["light"] * p["non-wear_flag"]

        p["masked_X"] = p["masked_X"].replace(0, np.nan)
        p["masked_Y"] = p["masked_Y"].replace(0, np.nan)
        p["masked_Z"] = p["masked_Z"].replace(0, np.nan)
        p["masked_enmo"] = p["masked_enmo"].replace(0, np.nan)
        p["masked_anglez"] = p["masked_anglez"].replace(0, np.nan)
        p["masked_light"] = p["masked_light"].replace(0, np.nan)
    
        mean_ = (
            p.groupby("relative_date_PCIAT")[self.scale_columns + self.masked_columns]
            .mean()
            .reset_index()
        )

        std_ = (
            p.groupby("relative_date_PCIAT")[self.scale_columns + self.masked_columns].std().reset_index()
        )

        p = mean_.merge(std_, on="relative_date_PCIAT", suffixes=("", "_mean")).drop(columns=["relative_date_PCIAT"])
        p = p.values
    
        scaler = StandardScaler()
        p = scaler.fit_transform(p)

        # nan -> 0
        p = np.nan_to_num(p)

        return {
            "input": torch.tensor(p, dtype=torch.float32),
            "output": torch.tensor(p, dtype=torch.float32),
        }
    
def read_parquet(base_dir, id_):
    path = os.path.join(base_dir, f"id={id_}", "part-0.parquet")
    return pd.read_parquet(path)

