import torch.nn as nn


def MLP(emd_dim, channel=64, hidden_size=1024):
    return nn.Sequential(
        nn.Linear(emd_dim, hidden_size),
        nn.BatchNorm1d(channel),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, emd_dim),
    )
