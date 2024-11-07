import numpy as np
import torch
import torch.nn as nn


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
        self.linear = nn.Linear(table_input_size+32, 1)

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

        self.downsample = nn.Sequential(
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

        self.pos_enc = PositionalEncoding(d_model=32, dropout=0.0, max_len=310)
        self.multihead_attention = MultiHeadAttention(embed_dim=32*10, num_heads=4, dropout=0.0)
        
    def forward(self, active_logs, active_mask):
        # active_logs: (1, 31, 17280, 15)
        # assert active_logs.shape[-1] == 15
        subsample_input = active_logs.permute(0, 3, 1, 2) # (1, 15, 31, 17280) # (batch, channel, day, time)
        subsample_input = subsample_input.reshape(1, 15, -1) # (1, 15, 31*17280) # (batch, channel, day*time)
        subsampled = self.downsample(subsample_input) # (1, 32, 310) # (batch, channel, day*time)
        # (1, 32, 310)
        
        # timeseries = subsampled.reshape(1, 32, 31, -1) # (1, 32, 31, 10) # (batch, channel, day, time)
        # timeseries = timeseries.permute(0, 2, 3, 1) # (1, 31, 10, 32) # (batch, day, time, channel)
        timeseries = subsampled # (1, 32, 310) # (batch, channel, day*time)
        timeseries = timeseries.permute(0, 2, 1) # (1, 310, 32) # (batch, day*time, channel)

        embedded, _ = self.encoder(timeseries) # x: (seq_len, batch, input_size), _: (h_n, c_n)
        embedded = self.pos_enc(embedded)
        # print(embedded.shape) # 1, 310, 32 # (batch, day*time, channel)

        embedded, attention_weights = self.multihead_attention(embedded, active_mask) # (1, 31, 320) # (batch, day, time*channel)
        pooled_embedded = embedded.view(1, 10, 32)
        pooled_embedded = pooled_embedded.mean(dim=1) # (1, 32) # (batch, channel)

        return pooled_embedded, embedded, attention_weights


class TimeAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(TimeAutoEncoder, self).__init__()

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

        self.downsample = nn.Sequential(
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

        self.pos_enc = PositionalEncoding(d_model=32, dropout=0.0, max_len=310)
        self.multihead_attention = MultiHeadAttention(embed_dim=32*10, num_heads=4, dropout=0.0)

        self.decoder = Decoder()

    def forward(self, active_logs, active_mask):
        # active_logs: (1, 31, 17280, 15)
        # assert active_logs.shape[-1] == 15
        subsample_input = active_logs.permute(0, 3, 1, 2) # (1, 15, 31, 17280) # (batch, channel, day, time)
        subsample_input = subsample_input.reshape(1, 15, -1) # (1, 15, 31*17280) # (batch, channel, day*time)
        subsampled = self.downsample(subsample_input) # (1, 32, 310) # (batch, channel, day*time)
        # (1, 32, 310)
        
        # timeseries = subsampled.reshape(1, 32, 31, -1) # (1, 32, 31, 10) # (batch, channel, day, time)
        # timeseries = timeseries.permute(0, 2, 3, 1) # (1, 31, 10, 32) # (batch, day, time, channel)
        timeseries = subsampled # (1, 32, 310) # (batch, channel, day*time)
        timeseries = timeseries.permute(0, 2, 1) # (1, 310, 32) # (batch, day*time, channel)

        embedded, _ = self.encoder(timeseries) # x: (seq_len, batch, input_size), _: (h_n, c_n)
        embedded = self.pos_enc(embedded)
        # print(embedded.shape) # 1, 310, 32 # (batch, day*time, channel)

        embedded, attention_weights = self.multihead_attention(embedded, active_mask) # (1, 31, 320) # (batch, day, time*channel)
        pooled_embedded = embedded#  * active_mask.unsqueeze(-1) # (1, 31, 320) # (batch, day, time*channel)
        pooled_embedded = pooled_embedded # (1, 1, 320) # (batch, 1, time*channel)

        pooled_embedded = pooled_embedded.unsqueeze(1) 

        embedded = embedded.view(1, 31, 10, 32) # (1, 31, 10, 32) # (batch, day, time, channel)

        input_ = torch.cat([pooled_embedded, embedded], dim=-1) # (1, 31, 10, 64) # (batch, day, time, channel*2)

        output = self.decoder(input_) # (1, 31, 17280, 15) # (batch, day, time, channel)

        return output, pooled_embedded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

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
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(
            64, 64, kernel_size=4, stride=2, padding=1
        )  # 時間方向に拡大
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(32, 15, kernel_size=4, stride=2, padding=1)

        self.upsample = nn.Upsample(size=17280, mode="linear")  # 最終的な長さに調整

    def forward(self, x):
        batch, days, time, channels = x.shape
        x = x.view(batch * days, channels, time)  # (batch*days, 32, 10)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        x = self.upsample(x)  # (batch*days, 15, 17280)
        x = x.view(batch, days, 17280, 15)

        return x