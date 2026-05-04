import torch
import torch.nn as nn

class ResidualPolicy(nn.Module):
    def __init__(self, obs_dim, n_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_assets)
        )

    def forward(self, obs):
        return torch.tanh(self.net(obs))