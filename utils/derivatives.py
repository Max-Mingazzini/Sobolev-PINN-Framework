from __future__ import annotations
from typing import Callable

import torch
from torch.func import jacfwd, vmap


__all__ = ["J_t_model", "J_p_model", "J_tk_model"]


def _single_J_t(model: Callable, t1: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
    """
    Per-sample time Jacobian: J_t y(t1, p1) -> [S]
    t1: [1], p1: [P]
    """

    def g_t(t_scalar: torch.Tensor) -> torch.Tensor:
        return model(t_scalar.view(1, 1), p1.view(1, -1)).squeeze(0)  # [S]

    return jacfwd(g_t)(t1.squeeze(0))  # [S]


def _single_J_p(model: Callable, t1: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
    """
    Per-sample parameter Jacobian: J_p y(t1, p1) -> [S, P]
    t1: [1], p1: [P]
    """

    def g_p(p_vec: torch.Tensor) -> torch.Tensor:
        return model(t1.view(1, 1), p_vec.view(1, -1)).squeeze(0)  # [S]

    return jacfwd(g_p)(p1)  # [S, P]


def _single_J_tk(model: Callable, t1: torch.Tensor, p1: torch.Tensor, order: int) -> torch.Tensor:
    """
    Per-sample k-th time derivative at (t1, p1) -> [S]
    order ≥ 1. For scalar input, repeated jacfwd returns a vector [S].
    """
    if order < 1:
        raise ValueError("order must be ≥ 1")

    def g(t_scalar: torch.Tensor) -> torch.Tensor:
        return model(t_scalar.view(1, 1), p1.view(1, -1)).squeeze(0)  # [S]

    fn = g
    for _ in range(order):
        fn = jacfwd(fn)
    return fn(t1.squeeze(0))  # [S]


def J_t_model(model: Callable, t_B1: torch.Tensor, p_BP: torch.Tensor) -> torch.Tensor:
    """
    Batched time Jacobian: J_t  y(t, p) for each batch item.
    Inputs : t_B1:[B,1], p_BP:[B,P]
    Output : [B, S]
    """
    return vmap(_single_J_t, in_dims=(None, 0, 0))(model, t_B1, p_BP)  # [B, S]


def J_p_model(model: Callable, t_B1: torch.Tensor, p_BP: torch.Tensor) -> torch.Tensor:
    """
    Batched parameter Jacobian: J_p y(t, p) for each batch item.
    Inputs : t_B1:[B,1], p_BP:[B,P]
    Output : [B, S, P]
    """
    return vmap(_single_J_p, in_dims=(None, 0, 0))(model, t_B1, p_BP)  # [B, S, P]


def J_tk_model(model: Callable, t_B1: torch.Tensor, p_BP: torch.Tensor, order: int) -> torch.Tensor:
    """
    Batched k-th time derivative for k ≥ 1.
    Inputs : t_B1:[B,1], p_BP:[B,P], order:int
    Output : [B, S]
    """
    return vmap(_single_J_tk, in_dims=(None, 0, 0, None))(model, t_B1, p_BP, order)  # [B, S]
