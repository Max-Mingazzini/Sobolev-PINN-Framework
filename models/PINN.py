import torch
import torch.nn as nn
from typing import List


class PINN(nn.Module):
    """
    Minimal PINN: (t, p) -> y
      t : [B, 1]
      p : [B, P]
      y : [B, S]
    """

    def __init__(self, param_dim: int, state_dim: int, hidden_layers: List[int] = [64, 64]):
        super().__init__()
        in_dim = 1 + int(param_dim)  # time and parameters
        dims = [in_dim] + list(hidden_layers) + [int(state_dim)]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on the last layer
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t, params], dim=-1)  # [B, 1+P]
        return self.net(x)  # [B, S]
