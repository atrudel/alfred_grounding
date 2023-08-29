from typing import Tuple

from torch import nn, Tensor


class VariationalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.fc_means = nn.Linear(hidden_size, z_size)
        self.fc_logvars = nn.Linear(hidden_size, z_size)

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        x = self.mlp(x)
        means = self.fc_means(x)
        log_vars = self.fc_logvars(x)
        return means, log_vars
