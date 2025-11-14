import torch
from .ODE import ODE, register_ode


@register_ode("lotka_volterra")
class LotkaVolterra(ODE):
    def f(self, t: torch.Tensor, v: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        alpha = params[..., 0]
        beta = params[..., 1]
        gamma = params[..., 2]
        delta = params[..., 3]
        x = v[..., 0]
        y = v[..., 1]
        dx_dt = alpha * x - beta * x * y
        dy_dt = -gamma * y + delta * x * y

        return torch.stack((dx_dt, dy_dt), dim=-1)
