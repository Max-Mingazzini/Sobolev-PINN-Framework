import torch
from .ODE import ODE, register_ode


@register_ode("your_ode")
class your_ode(ODE):
    def f(self, t: torch.Tensor, v: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Create RHS f for the ODE, all other ODE methods are derived from this one. Assumes batched input.
        Shapes: t [Batch, 1], y [Batch, d_y], params [Batch, d_p] -> dy/dt [Batch, d_y]
        """
        pass
