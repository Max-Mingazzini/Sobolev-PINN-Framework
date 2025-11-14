import torch
from torch import nn
from abc import abstractmethod

from utils.derivatives import J_t_model

ODE_REGISTRY = {}


def register_ode(name):
    def decorator(cls):
        ODE_REGISTRY[name] = cls
        return cls

    return decorator


class ODE(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def f(self, t: torch.Tensor, y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        RHS of ODE f(t, y, p)
        With shapes: t, [Batch, 1], y [Batch, Dim(y)] and params shaped [Batch, Dim(p)] -> dy/dt [Batch, Dim(y)]
        """
        pass

    def residual(self, model, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Vector residual r = d/d_t y_hat - f(t, y_hat, p).
        Shapes:
          t:[B,1], p:[B,P] -> r:[B,S]
        """
        t = t.requires_grad_(True)
        y_hat = model(t, p)  # [B,S]
        Jt_hat = J_t_model(model, t, p)  # [B,S]
        return Jt_hat - self.f(t, y_hat, p)  # [B,S]

    def vector_field(self, t: float, y_flat: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        RHS of ODE f(t, y, params) for a single t
        Used as an adapter for torchdiffeq.odeint
        takes flat y of shape [D]
        returns flat dy/dt of shape [D]
        """
        y = y_flat.unsqueeze(0) if y_flat.dim() == 1 else y_flat
        p = params.unsqueeze(0) if params.dim() == 1 else params
        t_single = torch.tensor([[t]], dtype=y.dtype, device=y.device)
        dy_dt = self.f(t_single, y, p)
        return dy_dt.squeeze(0)
