import numpy as np
import torch
import torch.nn as nn
from icecream import ic


class CMIModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(CMIModel, self).__init__()
        self.timeencoder = TimeEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        table_input_size = 63
        self.table_encoder = nn.Sequential(
            nn.Linear(table_input_size, table_input_size*2),
            nn.ReLU(),
            nn.Linear(table_input_size*2, table_input_size*2),
            nn.ReLU(),
            nn.Linear(table_input_size*2, table_input_size)
        )
        self.linear = nn.Linear(table_input_size+3968, 1)

    def forward(self, table_input, time_input, active_mask):
        pooled_embedded, embedded, attention_weights = self.timeencoder(time_input, active_mask)
        table_encoded = self.table_encoder(table_input).squeeze(1)
        # ic(table_encoded.shape)
        x = torch.cat([pooled_embedded, table_encoded], dim=-1)
        x = self.linear(x)
        return x, attention_weights
        

class TimeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(TimeEncoder, self).__init__()

        # watch_day = 31
        # oneday_sample = 17280
        # time_of_day = 24
        # kernel_time = 3
        # kernel_size = (oneday_sample*watch_day) // (time_of_day // kernel_time)
        # stride = kernel_size // 2
        # padding = (kernel_size - stride) // 2

        # subsample-setting
        kernel_size = 12
        stride = kernel_size // 2
        padding = (kernel_size - stride) // 2

        # LayerNormを追加

        self.subsample = nn.Sequential(
            #nn.LayerNorm([15, 31*17280]),
            nn.Conv1d(in_channels=15, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.LayerNorm([2880]),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.LayerNorm([480]),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.LayerNorm([80]),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.LayerNorm([13]),
        )

        self.encoder = nn.LSTM(input_size=32, hidden_size=32, num_layers=num_layers, dropout=dropout, batch_first=True)

        
    def forward(self, active_logs, active_mask):
        # active_logs: (1, 31, 17280, 15)
        # assert active_logs.shape[-1] == 15
        subsample_input = active_logs.permute(0, 3, 1, 2) # (1, 15, 31, 17280) # (batch, channel, day, time)
        subsample_input = subsample_input.permute(0, 2, 1, 3).squeeze(0) # (31, 15, 17280) # (batch, channel, time)
        
        embedded = self.subsample(subsample_input)
        #ic(embedded.shape)
        embeeded_mean = embedded.mean(dim=-1)
        #ic(embeeded_mean.shape) # 31, 128
        reshape_embedded = embeeded_mean.reshape(1, -1) # 1, 3968
        #ic(reshape_embedded.shape)

        return reshape_embedded, embedded, None