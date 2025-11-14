import os
import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from torch.func import jacfwd, vmap
from torchdiffeq import odeint

from Equations.ODE import ODE_REGISTRY
from utils.generator import param_generator


def _as_tensor(x):
    dtype = torch.get_default_dtype()
    return torch.as_tensor(x, dtype=dtype)


def _ensure_tensor_list(x) -> torch.Tensor:
    """Ensure params are [N,P]"""
    t = torch.as_tensor(x, dtype=torch.get_default_dtype())
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def _hash_key(d: Dict) -> str:
    """short key for cache filenames"""
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _odeint_single_param(eq, y0: torch.Tensor, t_grid: torch.Tensor, p_i: torch.Tensor, odeint_options) -> torch.Tensor:
    """
    Performs a single solver run for a parameter on the ODE
    """
    dt = torch.get_default_dtype()
    y0 = y0.detach().to(dtype=torch.float64)
    t_points = (t_grid.detach().squeeze(-1) if t_grid.ndim == 2 else t_grid.detach()).to(dtype=torch.float64, device=y0.device)
    p_i = p_i.detach().to(dtype=torch.float64, device=y0.device)

    def rhs_state(t_scalar, y_vec):
        t_b = torch.as_tensor([[t_scalar]], dtype=torch.float64, device=y0.device)
        y_b = y_vec.view(1, -1)
        p_b = p_i.view(1, -1)
        dy_b = eq.f(t_b, y_b, p_b)
        return dy_b.squeeze(0)

    with torch.no_grad():
        y_TS = odeint(rhs_state, y0, t_points, **odeint_options)
    return y_TS.to(dtype=dt)


def _integrate_augmented_for_sens(
    eq, y0: torch.Tensor, t_grid: torch.Tensor, p_i: torch.Tensor, odeint_options
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a single solver run for a parameter on ODE and forward sensitivitites
    """
    dt = torch.get_default_dtype()
    y0 = y0.detach().to(dtype=torch.float64)
    t_points = (t_grid.detach().squeeze(-1) if t_grid.ndim == 2 else t_grid.detach()).to(dtype=torch.float64, device=y0.device)
    p_i = p_i.detach().to(dtype=torch.float64, device=y0.device)

    S_dim = y0.numel()
    P_dim = p_i.numel()
    S0 = torch.zeros(S_dim, P_dim, dtype=torch.float64, device=y0.device)
    z0 = torch.cat([y0.view(-1), S0.view(-1)], dim=0)

    def rhs_aug(t_scalar, z_vec):
        y = z_vec[:S_dim]
        S_matrix = z_vec[S_dim:].view(S_dim, P_dim)
        t_b = torch.as_tensor([[t_scalar]], dtype=torch.float64, device=y0.device)
        y_b = y.view(1, -1)
        p_b = p_i.view(1, -1)
        f_val = eq.f(t_b, y_b, p_b).squeeze(0)

        def g_y(y_vec):
            return eq.f(t_b, y_vec.view(1, -1), p_b).squeeze(0)

        def g_p(p_vec):
            return eq.f(t_b, y_b, p_vec.view(1, -1)).squeeze(0)

        J_y_f = jacfwd(g_y)(y)
        J_p_f = jacfwd(g_p)(p_i)
        dS_dt = J_y_f @ S_matrix + J_p_f
        return torch.cat([f_val, dS_dt.view(-1)], dim=0)

    with torch.no_grad():
        z_T = odeint(rhs_aug, z0, t_points, **odeint_options)

    y_TS = z_T[:, :S_dim].to(dtype=dt)
    S_TSP = z_T[:, S_dim:].view(-1, S_dim, P_dim).to(dtype=dt)
    return y_TS, S_TSP


def _time_derivs_targets(eq, t_grid, y_TS, p_i, max_order):
    """
    Returns the time derivatives used for temporal Sobolev training, up to order 2.
    """
    if max_order <= 0:
        return []
    if max_order > 2:
        raise NotImplementedError("time_order>2 not implemented")

    T = y_TS.shape[0]
    t_T1 = (t_grid if t_grid.ndim == 2 else t_grid.unsqueeze(-1)).to(dtype=y_TS.dtype, device=y_TS.device)
    p_TP = p_i.to(dtype=y_TS.dtype, device=y_TS.device).unsqueeze(0).expand(T, -1)

    with torch.no_grad():
        d1 = eq.f(t_T1, y_TS, p_TP)
    result = [d1]
    if max_order == 1:
        return result

    def J_y_f_single(t1, y1, p1):
        def g_y(y_vec):
            return eq.f(t1.view(1, 1), y_vec.view(1, -1), p1.view(1, -1)).squeeze(0)

        return jacfwd(g_y)(y1)

    def J_t_f_single(t1, y1, p1):
        def g_t(t_scalar):
            return eq.f(t_scalar.view(1, 1), y1.view(1, -1), p1.view(1, -1)).squeeze(0)

        return jacfwd(g_t)(t1.squeeze(0)).view(-1)

    J_y_f_TSS = vmap(J_y_f_single)(t_T1, y_TS, p_TP)
    J_t_f_TS = vmap(J_t_f_single)(t_T1, y_TS, p_TP)
    d2 = J_t_f_TS + (J_y_f_TSS @ d1.unsqueeze(-1)).squeeze(-1)
    result.append(d2)
    return result


@dataclass(frozen=True)
class CacheSpec:
    eq_name: str
    y0: Tuple[float, ...]
    t0: float
    t_final: float
    n_time: int
    p: Tuple[float, ...]
    need_sens: bool
    time_order: int
    dtype: str


def _cache_file(cache_dir: Path, spec: CacheSpec) -> Path:
    """
    Returns the path for a specific CacheSpec, where the cache is saved.
    """
    key = _hash_key(
        {
            "eq": spec.eq_name,
            "y0": spec.y0,
            "t0": spec.t0,
            "t1": spec.t_final,
            "T": spec.n_time,
            "p": spec.p,
            "sens": spec.need_sens,
            "k": spec.time_order,
            "dtype": spec.dtype,
        }
    )
    return cache_dir / f"{spec.eq_name}_{key}.pt"


def _load_cache(path: Path):
    if path.exists():
        return torch.load(path, map_location="cpu")
    return None


def _save_cache(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class ODEDataset(Dataset):
    """
    Flat dataset: each item is one (t, p, y) triple (+ optional time_targets, sens).
      t:  [1]
      p:  [P]
      y:  [S]
      time_targets: [K, S]  (present iff train.time_order > 0)
      sens:         [S, P]  (present iff train.use_param_sens)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        train_cfg = cfg["train"]
        data_cfg = cfg["data"]

        total_start_time = time.perf_counter()
        self.total_build_time = None
        self.was_cached = False

        # equation & cache root
        self.eq_name = cfg["equation"]["name"]
        self.eq = ODE_REGISTRY[self.eq_name]()
        self.cache_dir = Path(cfg.get("cache_dir", "cache")) / self.eq_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # toggles
        self.time_order = int(train_cfg.get("time_order", 0))
        self.use_sens = bool(train_cfg.get("use_param_sens", False))
        self.dtype_str = str(torch.get_default_dtype())

        # grid / IC / params
        self.T = int(data_cfg["n_time_points"])
        t0 = float(data_cfg["t0"])
        t1 = float(data_cfg["t_final"])
        self.t_grid = torch.linspace(t0, t1, self.T, dtype=torch.get_default_dtype())  # [T]
        self.y0 = _as_tensor(data_cfg["initial_condition"]).view(-1)  # [S]
        self.params = _ensure_tensor_list(param_generator(config=data_cfg))  # [N,P]
        N = self.params.shape[0]
        odeint_options = data_cfg.get("odeint_options", {})

        # dataset-level cache
        params_md5 = hashlib.md5(self.params.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
        ds_key = _hash_key(
            {
                "eq": self.eq_name,
                "y0": tuple(self.y0.tolist()),
                "t0": t0,
                "t1": t1,
                "T": self.T,
                "k": self.time_order,
                "sens": self.use_sens,
                "dtype": self.dtype_str,
                "params_md5": params_md5,
                "odeint": {
                    "method": odeint_options.get("method", None),
                    "rtol": odeint_options.get("rtol", None),
                    "atol": odeint_options.get("atol", None),
                    "step_size": odeint_options.get("step_size", None),
                },
            }
        )
        ds_cache = self.cache_dir / f"{self.eq_name}_dataset_{ds_key}.pt"
        payload = _load_cache(ds_cache)
        if payload is not None:
            self.t_flat = payload["t_flat"]
            self.p_flat = payload["p_flat"]
            self.y_flat = payload["y_flat"]
            self.time_targets_flat = payload["time_targets_flat"]
            self.sens_flat = payload["sens_flat"]
            meta = payload["meta"]
            self.N, self.S, self.P, self.T = (
                int(meta["N"]),
                int(meta["S"]),
                int(meta["P"]),
                int(meta["T"]),
            )

            self.was_cached = True
            self.total_build_time = float(meta.get("total_build_time", 0.0))
            return

        # compute once, flat tensors directly
        t_block = self.t_grid.view(self.T, 1)  # [T,1]
        y_list: List[torch.Tensor] = []
        tt_list: Optional[List[torch.Tensor]] = [] if self.time_order > 0 else None
        s_list: Optional[List[torch.Tensor]] = [] if self.use_sens else None

        t_start = time.perf_counter()
        for i in range(N):
            if i % 100 == 0 and i > 0:
                dt = time.perf_counter() - t_start
                avg = dt / i
                estimate = (N - i) * avg
                print(f"[Dataset] {i}/{N} params | {avg:.1f}s/param | remaining {estimate/60:.1f} min")

            p_i = self.params[i]  # [P]

            # state with otpional sensitivities
            if self.use_sens:
                y_TS, S_TSP = _integrate_augmented_for_sens(self.eq, self.y0, self.t_grid, p_i, odeint_options)
            else:
                y_TS = _odeint_single_param(self.eq, self.y0, self.t_grid, p_i, odeint_options)
                S_TSP = None

            # time-derivative targets up to K=time_order
            if self.time_order > 0:
                d_list = _time_derivs_targets(self.eq, self.t_grid, y_TS, p_i, self.time_order)  # list of [T,S]
                time_targets = torch.stack(d_list, dim=0)  # [K,T,S]
            else:
                time_targets = None

            # collect flat blocks
            y_list.append(y_TS)  # [T,S]
            if tt_list is not None and time_targets is not None:
                tt_list.append(time_targets.permute(1, 0, 2))  # [T,K,S]
            if s_list is not None and S_TSP is not None:
                s_list.append(S_TSP)  # [T,S,P]

        # finalize flats
        # not memory efficient, with issues in scaling this section should be refactored
        self.y_flat = torch.cat(y_list, dim=0)  # [N*T, S]
        self.t_flat = t_block.expand(N, self.T, 1).reshape(-1, 1)  # [N*T, 1]
        self.p_flat = torch.repeat_interleave(self.params, repeats=self.T, dim=0)  # [N*T, P]
        self.time_targets_flat = None if tt_list is None else torch.cat(tt_list, dim=0)  # [N*T, K, S]
        self.sens_flat = None if s_list is None else torch.cat(s_list, dim=0)  # [N*T, S, P]

        # meta data
        self.N = N
        self.S = self.y_flat.shape[-1]
        self.P = self.p_flat.shape[-1]

        self.total_build_time = time.perf_counter() - total_start_time

        # write dataset-level cache for next runs
        _save_cache(
            ds_cache,
            {
                "t_flat": self.t_flat,
                "p_flat": self.p_flat,
                "y_flat": self.y_flat,
                "time_targets_flat": self.time_targets_flat,
                "sens_flat": self.sens_flat,
                "meta": {
                    "N": self.N,
                    "S": self.S,
                    "P": self.P,
                    "T": self.T,
                    "total_build_time": self.total_build_time,
                },
            },
        )

    def __len__(self) -> int:
        return self.N * self.T

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {"t": self.t_flat[idx], "p": self.p_flat[idx], "y": self.y_flat[idx]}
        if self.time_targets_flat is not None:
            out["time_targets"] = self.time_targets_flat[idx]  # [K, S]
        if self.sens_flat is not None:
            out["sens"] = self.sens_flat[idx]  # [S, P]
        return out


def create_dataloaders(cfg: dict, dataset: ODEDataset | None = None) -> Tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders"""
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 2048))
    shuffle = bool(train_cfg.get("shuffle", True))
    test_share = float(train_cfg.get("test_split_share", 0.2))
    samples_per_epoch = int(train_cfg.get("samples_per_epoch", 0))

    if dataset is None:
        dataset = ODEDataset(cfg)

    N = len(dataset)
    test_size = int(round(test_share * N))
    train_size = N - test_size

    train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1337))

    dl_args = dict(batch_size=batch_size, drop_last=False, num_workers=4, persistent_workers=False)

    sampler_train = None
    if samples_per_epoch > 0:
        sampler_train = RandomSampler(
            train_set,
            replacement=True,
            num_samples=samples_per_epoch,
            generator=torch.Generator().manual_seed(1337),
        )

    if sampler_train is None:
        train_loader = DataLoader(train_set, shuffle=shuffle, **dl_args)
    else:
        train_loader = DataLoader(train_set, sampler=sampler_train, **dl_args)
    val_loader = DataLoader(test_set, **dl_args)
    return train_loader, val_loader
