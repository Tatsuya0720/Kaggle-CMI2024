import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

class CMIDataset(Dataset):
    def __init__(self, table_df, valid_ids, base_dir):
        self.base_dir = base_dir
        self.table_df = table_df
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
        # テーブルデータの抽出
        id_ = self.valid_ids[idx]

        save_dir = "/home/tatsuya/code/projects/kaggle/ChildMindInstitute2024/precreated_dataset/exp010/"
        save_path = os.path.join(save_dir, id_)
    
        if not os.path.exists(save_path):
            table = self.table_df.loc[self.table_df["id"]==self.valid_ids[idx], :]
            table_feature = table.drop(columns=["id", "sii"]).values
            sii = table["sii"].values

            # 時系列データの抽出
            p = read_parquet(self.base_dir, self.valid_ids[idx])

            scaler_features = p[self.scale_columns].values
            scaler = StandardScaler()
            p[self.scale_columns] = scaler.fit_transform(scaler_features)

            for mask_col in self.masked_columns:
                p[mask_col] = p[mask_col.replace("masked_", "")] * p["non-wear_flag"]

            use_cols = self.masked_columns + self.original_columns + self.scale_columns
            # p = np.nan_to_num(p)
            p = p.fillna(0)

            # to chunk
            groups = p.groupby("relative_date_PCIAT")
            # グループごとにデータフレームのリストに分割
            chunks = [group.reset_index(drop=True) for _, group in groups]

            watch_day = len(chunks)
            active_logs = np.zeros((31, 17280, len(use_cols)), dtype=np.float32)
            active_mask = np.zeros((31), dtype=np.int32)

            for i, chunk in enumerate(chunks):
                if i==0: # 
                    active_logs[i, -len(chunk):, :] = chunk[use_cols].values
                elif i==watch_day:
                    active_logs[i, :len(chunk), :] = chunk[use_cols].values
                else:
                    array = chunk[use_cols].values
                    active_logs[i, :len(array), :] = array

                active_mask[i] = 1

                if i==30:
                    break

            # active_logs: (31, 17280, 9)

            dataset_ = {
                "table_input": torch.tensor(table_feature, dtype=torch.float32),
                "time_input": torch.tensor(active_logs, dtype=torch.float32),
                "mask": torch.tensor(active_mask, dtype=torch.int32),
                "output": torch.tensor(sii, dtype=torch.float32),
            }

            os.makedirs(save_path)
            with open(os.path.join(save_path, "dataset.pkl"), "wb") as write_file:
                pickle.dump(dataset_, write_file)
        else: # 読み込み
            with open(os.path.join(save_path, "dataset.pkl"), "rb") as read_file:
                dataset_ = pickle.load(read_file)

        return dataset_
    
def read_parquet(base_dir, id_):
    path = os.path.join(base_dir, f"id={id_}", "part-0.parquet")
    return pd.read_parquet(path)

