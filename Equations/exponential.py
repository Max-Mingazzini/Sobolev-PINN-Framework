import torch
from .ODE import ODE, register_ode


@register_ode("exponential")
class Exponential(ODE):
    def f(self, t: torch.Tensor, v: torch.Tensor, params: torch.Tensor):
        k = params[..., 0]
        y = v[..., 0]
        dy_dt = k * y
        return dy_dt
