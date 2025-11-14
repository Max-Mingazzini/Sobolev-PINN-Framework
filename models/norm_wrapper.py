# models/norm_wrapper.py
import torch
import torch.nn as nn


class NormedPINN(nn.Module):
    """
    Wraps a base PINN that expects normalized inputs and outputs normalized y.
    This wrapper:
      - normalizes (t, p) from physical -> normalized
      - runs base(t_norm, p_norm)
      - de-normalizes y back to physical units
    """

    def __init__(
        self,
        base: nn.Module,
        t0: float,
        t_scale: float,
        p0: torch.Tensor,
        p_scale: torch.Tensor,  # [P]
        y_shift: torch.Tensor = None,  # [S] or scalar
        y_scale: torch.Tensor = None,  # [S] or scalar
        dtype=None,
    ):
        super().__init__()
        self.base = base
        dt = torch.get_default_dtype() if dtype == None else dtype

        self.register_buffer("t0", torch.as_tensor([t0], dtype=dt))  # [1]
        self.register_buffer("t_scale", torch.as_tensor([t_scale], dtype=dt))  # [1]

        p0 = torch.as_tensor(p0, dtype=dt).view(1, -1)  # [1,P] for broadcasting
        p_scale = torch.as_tensor(p_scale, dtype=dt).view(1, -1)  # [1,P]
        self.register_buffer("p0", p0)
        self.register_buffer("p_scale", p_scale)

        # defaults: no output normalization if not provided
        if y_shift is None:
            y_shift = 0.0
        if y_scale is None:
            y_scale = 1.0
        y0 = torch.as_tensor(y_shift, dtype=dt).view(1, -1)  # [1,S]
        y_scale = torch.as_tensor(y_scale, dtype=dt).view(1, -1)  # [1,S]
        self.register_buffer("y_shift", y0)
        self.register_buffer("y_scale", y_scale)

    def forward(self, t_phys: torch.Tensor, p_phys: torch.Tensor) -> torch.Tensor:
        # normalize inputs
        t_norm = (t_phys - self.t0) / self.t_scale  # [B,1], scale to [0,1]
        p_norm = ((p_phys - self.p0) / self.p_scale) * 2  # [B,P], scale to [-1, 1]

        # base model returns normalized y, denormalize to physical
        y_norm = self.base(t_norm, p_norm)  # [B,S]
        y_phys = self.y_shift + self.y_scale * y_norm  # [B,S]
        return y_phys
