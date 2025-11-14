import os, importlib, random, pkgutil, json
from datetime import datetime
from pathlib import Path
import argparse

import yaml
import numpy as np
import torch

from models.PINN import PINN
from models.norm_wrapper import NormedPINN
from Equations.ODE import ODE_REGISTRY
from training.dataloader import ODEDataset
import training.dataloader as dataloader
from training.trainer import Trainer


def setup_torch(dtype_str=None, device_str=None, seed=42):
    # dtype
    dt = torch.float64 if (str(dtype_str).lower() == "float64") else torch.float32
    torch.set_default_dtype(dt)
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(1)

    # device
    if device_str:
        d = device_str.lower()
        if d.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(d if ":" in d else "cuda")
        elif d == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif d == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("[run] dtype:", torch.get_default_dtype(), "| device:", device)
    return device


def auto_import_equations(pkg_name: str):
    pkg = importlib.import_module(pkg_name)
    if not hasattr(pkg, "__path__"):
        return
    for mod in pkgutil.iter_modules(pkg.__path__):
        name = mod.name
        if name == "ODE":
            continue
        importlib.import_module(f"{pkg_name}.{name}")


def validate_cfg(cfg: dict):
    train = cfg.get("train", cfg.get("training", {}))
    for k in ["equation", "model"]:
        if k not in cfg:
            raise KeyError(f"config missing top-level key: '{k}'")
    if "name" not in cfg["equation"]:
        raise KeyError("config['equation']['name'] required (e.g., 'lotka_volterra')")
    t = cfg["train"]
    t.setdefault("epochs", 2000)
    t.setdefault("learning_rate", 1e-3)
    t.setdefault("weight_decay", 0.0)
    t.setdefault("batch_size", 256)
    t.setdefault("shuffle", True)
    t.setdefault("test_split_share", 0.2)
    t.setdefault("w_phys", 1.0)
    t.setdefault("w_data", 0.0)
    t.setdefault("w_time", 0.0)
    t.setdefault("w_param", 0.0)
    t.setdefault("time_order", 0)
    t.setdefault("log_interval", 50)
    t.setdefault("clip_grad", 0.0)
    t.setdefault("use_param_sens", False)
    t.setdefault("compile", False)
    return cfg


def make_run_dir(root="runs", tag="experiment", explicit=None):
    """
    Returns the path of and creates the directory where paths are stored
    """
    if explicit is not None:
        run_dir = Path(explicit)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def infer_y_scaling_from_dataset(
    cfg,
    max_samples=100_000,
    q_low=0.01,
    q_high=0.99,
    epsilon=1e-6,
    seed=1337,
    ds: ODEDataset | None = None,
):
    """
    Returns the scale used to build the denormalization layer of the NN
    """
    # build only if not provided
    if ds is None:
        ds = ODEDataset(cfg)
    y = ds.y_flat  # [N*T, S]

    n = y.shape[0]
    if n > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_samples]
        y = y.index_select(0, idx)

    q = torch.tensor([q_low, q_high], dtype=y.dtype)
    y_lo, y_hi = torch.quantile(y, q, dim=0)
    y_shift = 0.5 * (y_lo + y_hi)
    y_scale = torch.clamp(0.5 * (y_hi - y_lo), min=epsilon)
    return (
        y_shift.to(torch.get_default_dtype()),
        y_scale.to(torch.get_default_dtype()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = validate_cfg(cfg)

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    seed = int(train_cfg.get("seed", 42))
    device = setup_torch(
        dtype_str=train_cfg.get("dtype", None),
        device_str=train_cfg.get("device", None),
        seed=seed,
    )

    auto_import_equations("Equations")

    eq_name = cfg["equation"]["name"]
    if eq_name not in ODE_REGISTRY:
        available = ", ".join(sorted(ODE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown equation '{eq_name}'. Available: [{available}]. "
            "Make sure the module exists and registers with @register_ode."
        )
    ode = ODE_REGISTRY[eq_name]()

    # Build dataset
    print("[run] building/reading dataset cache")
    ds = ODEDataset(cfg)
    print(f"[run] dataset ready (cached = {ds.was_cached}) in {ds.total_build_time}")
    y_shift, y_scale = infer_y_scaling_from_dataset(cfg, ds=ds)
    print(f"[run] y-scaling ready: shift={y_shift.tolist()} scale={y_scale.tolist()}")

    # build the model with scale buffers
    base = PINN(**cfg["model"]).to(device=device, dtype=torch.get_default_dtype())
    t0, t_final = data_cfg["t0"], data_cfg["t_final"]
    t_scale = t_final - t0

    ranges = data_cfg.get("param_ranges")
    if ranges:
        p0 = [(low + high) / 2 for (low, high) in ranges]
        p_scale = [(high - low) for (low, high) in ranges]
    else:
        P = cfg["model"]["param_dim"]
        p0 = [0.0] * P
        p_scale = [1.0] * P

    model = NormedPINN(base, t0, t_scale, p0, p_scale, y_shift, y_scale).to(device=device, dtype=torch.get_default_dtype())

    if train_cfg.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    run_dir = make_run_dir(
        root=cfg.get("run_root", "runs"),
        tag=cfg.get("tag", eq_name),
        explicit=cfg.get("run_dir"),
    )

    with open(run_dir / "config_effective.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("[run] creating dataloaders...")
    train_loader, test_loader = dataloader.create_dataloaders(cfg, dataset=ds)
    print("[run] dataloaders ready. starting training...")
    trainer = Trainer(model=model, ode=ode, cfg=cfg)
    trainer = trainer.train(loader=train_loader, val_loader=test_loader)

    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(
            {
                "dataset_build_time": float(ds.total_build_time),
                "train_loop_time": float(trainer.total_train_time),
                "dataset_cached": bool(ds.was_cached),
                "N": ds.N,
                "T": ds.T,
                "P": ds.P,
            },
            f,
            indent=2,
        )

    with open(run_dir / "loss_statistics.json", "w") as f:
        json.dump(trainer.history, f, sort_keys=False)

    final_path = run_dir / "pinn_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"[done] saved final model to: {final_path}")
