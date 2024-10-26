import numpy as np
import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x, mode='encode-decode'):
        x, _ = self.encoder(x) # x: (seq_len, batch, input_size), _: (h_n, c_n)
        if mode == 'encode-decode':
            x, _ = self.decoder(x) # x: (seq_len, batch, input_size), _: (h_n, c_n)
        return x