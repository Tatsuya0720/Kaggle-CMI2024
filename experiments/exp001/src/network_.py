import numpy as np
import torch
import torch.nn as nn


class TableAutoEncoder(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(TableAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_items),
        )

    def forward(self, x, mode='encode-decode'):
        x = self.encoder(x)
        if mode == 'encode':
            return x
        
        elif mode == 'encode-decode':
            x = self.decoder(x)
            return x
        raise ValueError(f'Invalid mode: {mode}')