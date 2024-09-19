import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(sizes[1], sizes[2])
        )

    def forward(self, x):
        return self.model(x)
