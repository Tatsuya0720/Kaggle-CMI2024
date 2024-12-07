import numpy as np
import torch
import torch.nn as nn


class CMIModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(CMIModel, self).__init__()
        self.timeencoder = TimeEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        table_input_size = 68
        self.table_encoder = nn.Sequential(
            nn.Linear(table_input_size, table_input_size*2),
            nn.ReLU(),
            nn.LayerNorm(table_input_size*2),
            nn.Linear(table_input_size*2, table_input_size*2),
            nn.ReLU(),
            nn.LayerNorm(table_input_size*2),
            nn.Linear(table_input_size*2, table_input_size*2),
            nn.ReLU(),
            nn.LayerNorm(table_input_size*2),
            nn.Linear(table_input_size*2, table_input_size)
        )
        self.linear = nn.Linear(table_input_size+31+134, 22)

    def forward(self, table_input, time_input, active_mask):
        v_mean, h_mean, _ = self.timeencoder(time_input, active_mask)
        table_encoded = self.table_encoder(table_input).squeeze(1)
        # print(pooled_embedded.shape, table_encoded.shape)
        x = torch.cat([v_mean, h_mean, table_encoded], dim=-1)
        x = self.linear(x) # (1, 22)

        survey_pred = x[:, :21] # (1, 21)
        survey_pred = torch.sigmoid(survey_pred)
        sii_pred = x[:, -1].unsqueeze(0) # (1, 1)

        assert sii_pred.shape == (1, 1)
        assert survey_pred.shape == (1, 21)

        return sii_pred, survey_pred, None
        

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
        kernel_size = 5
        stride = kernel_size // 2
        padding = (kernel_size - stride) // 2

        # LayerNormを追加

        self.subsample = nn.Sequential(
            #nn.LayerNorm([15, 31*17280]),
            nn.Conv1d(in_channels=31*15, out_channels=31*15*2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.LayerNorm([31*15*2, 8639]),
            nn.Conv1d(in_channels=31*15*2, out_channels=31*15*1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=31*15*1, out_channels=31*10, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=31*10, out_channels=31*5, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=31*5, out_channels=31*1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=31*1, out_channels=31*1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=31*1, out_channels=31*1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),

        )
        
    def forward(self, active_logs, active_mask):
        # active_logs: (1, 31, 17280, 15)
        active_logs = active_logs.permute(0, 1, 3, 2) # (1, 31, 15, 17280)
        active_logs = active_logs.reshape(1, active_logs.size(1)*active_logs.size(2), active_logs.size(3)) # (1, 31*15, 17280)

        #print("input", active_logs.shape)
        subsampled = self.subsample(active_logs) # (1, 31, 134)
        #print("subsampled", subsampled.shape)

        v_mean = subsampled.mean(dim=-1, keepdim=True).squeeze(-1) # (1, 31)
        h_mean = subsampled.mean(dim=1, keepdim=True).squeeze(1) # (1, 134)

        return v_mean, h_mean, None

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """(batch, day, time, channel)の4次元テンソルが入力された時，dayごとにattentionを計算する．

        Args:
            embed_dim (_type_): _description_
            num_heads (_type_): _description_
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, x, mask=None):
        # x: (batch, day, time, channel)
        x = x.view(1, 31, -1) # (batch, day, time*channel)
        # query用に1.0のみの行列を作成
        query = torch.ones_like(x)[:, :1, :]
        x, attention_weights = self.attention(query=query, key=x, value=x, attn_mask=mask)
        return x, attention_weights