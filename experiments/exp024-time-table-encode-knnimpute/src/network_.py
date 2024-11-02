import numpy as np
import torch
import torch.nn as nn


class CMIModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(CMIModel, self).__init__()
        self.timeencoder = TimeEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        table_input_size = 97 + 15
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
        self.linear = nn.Linear(table_input_size+320, 1)

    def forward(self, table_input, time_input, active_mask):
        pooled_embedded, embedded, attention_weights = self.timeencoder(time_input, active_mask)
        table_encoded = self.table_encoder(table_input).squeeze(1)
        # print(pooled_embedded.shape, table_encoded.shape)
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
        kernel_size = 24
        stride = kernel_size // 2
        padding = (kernel_size - stride) // 2

        # LayerNormを追加

        self.subsample = nn.Sequential(
            #nn.LayerNorm([15, 31*17280]),
            nn.Conv1d(in_channels=15, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.LayerNorm([32, 44640]),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.LayerNorm([32, 3720]),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.LayerNorm([32, 310]),
        )

        self.encoder = nn.LSTM(input_size=32, hidden_size=32, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.multihead_attention = MultiHeadAttention(embed_dim=32*10, num_heads=4, dropout=0.0)
        
    def forward(self, active_logs, active_mask):
        # active_logs: (1, 31, 17280, 15)
        # assert active_logs.shape[-1] == 15
        subsample_input = active_logs.permute(0, 3, 1, 2) # (1, 15, 31, 17280) # (batch, channel, day, time)
        subsample_input = subsample_input.reshape(1, 15, -1) # (1, 15, 31*17280) # (batch, channel, day*time)
        subsampled = self.subsample(subsample_input) # (1, 32, 310) # (batch, channel, day*time)
        # (1, 32, 310)
        
        # timeseries = subsampled.reshape(1, 32, 31, -1) # (1, 32, 31, 10) # (batch, channel, day, time)
        # timeseries = timeseries.permute(0, 2, 3, 1) # (1, 31, 10, 32) # (batch, day, time, channel)
        timeseries = subsampled # (1, 32, 310) # (batch, channel, day*time)
        timeseries = timeseries.permute(0, 2, 1)

        embedded, _ = self.encoder(timeseries) # x: (seq_len, batch, input_size), _: (h_n, c_n)

        embedded, attention_weights = self.multihead_attention(embedded, active_mask) # (1, 31, 320) # (batch, day, time*channel)
        pooled_embedded = embedded#  * active_mask.unsqueeze(-1) # (1, 31, 320) # (batch, day, time*channel)
        pooled_embedded = pooled_embedded.squeeze(1)

        return pooled_embedded, embedded, attention_weights

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