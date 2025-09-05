import torch
import torch.nn as nn


class ProprioEncoder(nn.Module):
    def __init__(self, in_chans, emb_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        return self.mlp(x)