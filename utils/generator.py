import torch


def _get_ranges(cfg):
    """
    Return validated ragnes for parameters (low, high) from config
    """
    rngs = cfg.get("param_ranges", cfg.get("param_constraints"))
    if not rngs:
        raise ValueError("Need 'param_ranges' or 'param_constraints'.")
    out = []
    for pair in rngs:
        if len(pair) != 2:
            raise ValueError(f"Each range must be [low, high], got {pair}.")
        lo, hi = float(pair[0]), float(pair[1])
        if not (hi > lo):
            raise ValueError(f"Range must satisfy hi>lo, got [{lo}, {hi}].")
        out.append((lo, hi))
    return out


def _grid(ranges, n_per_dim):
    """
    Creates a parameter grid for the specified ranges
    """
    if n_per_dim <= 0:
        raise ValueError("n_params_per_dimension must be >= 1.")
    dt = torch.get_default_dtype()
    spaces = [torch.linspace(lo, hi, n_per_dim, dtype=dt) for (lo, hi) in ranges]
    return torch.cartesian_prod(*spaces)  # [N, P]


def _random_uniform(ranges, n_samples, seed=None):
    "Randomly samples the parameters from a uniform distribution with the specified ranges"
    if n_samples <= 0:
        raise ValueError("n_param_samples must be >= 1.")
    dt = torch.get_default_dtype()
    P = len(ranges)
    g = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    low = torch.tensor([r[0] for r in ranges], dtype=dt)  # [P]
    high = torch.tensor([r[1] for r in ranges], dtype=dt)  # [P]
    u = torch.rand((n_samples, P), dtype=dt, generator=g)  # [N,P]
    return low + (high - low) * u


def param_generator(config):
    """
    Returns [N, P] parameters on CPU.
    """
    mode = config.get("parameter_mode", config.get("param_mode", "random")).lower()
    ranges = _get_ranges(config)

    if mode == "grid":
        n = int(config.get("n_params_per_dimension", 5))
        params = _grid(ranges, n)
    elif mode in ("random", "uniform"):
        n = int(config.get("n_param_samples", 64))
        seed = config.get("seed", None)
        params = _random_uniform(ranges, n, seed)
    elif mode == "list":
        params = torch.as_tensor(config["param_list"], dtype=torch.get_default_dtype())
        if params.ndim == 1:
            params = params.unsqueeze(0)
        if params.shape[-1] != len(ranges):
            raise ValueError(f"param_list last dim {params.shape[-1]} != number of parameters {len(ranges)}.")
    else:
        raise ValueError(f"Unknown parameter_mode '{mode}'. Use 'grid' or 'random'.")

    params = torch.as_tensor(params, dtype=torch.get_default_dtype(), device="cpu")
    if params.ndim != 2:
        params = params.view(-1, params.shape[-1])
    return params
