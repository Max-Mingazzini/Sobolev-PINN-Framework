import torch
from .ODE import ODE, register_ode


@register_ode("generalized_lotka_volterra")
class GeneralizedLotkaVolterra(ODE):
    def f(self, t: torch.Tensor, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if t.dtype != x.dtype or t.device != x.device:
            t = t.to(dtype=x.dtype, device=x.device)
        if params.dtype != x.dtype or params.device != x.device:
            params = params.to(dtype=x.dtype, device=x.device)

        B, S = x.shape
        P = params.shape[-1]
        expected = S + S * S
        if P != expected:
            raise ValueError(f"[GLV] params dim {P} != S+S^2={expected} for S={S}")

        r = params[:, :S]  # [B, S]
        A = params[:, S:].view(B, S, S)  # [B, S, S]
        Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)  # [B, S]
        return x * (r + Ax)
