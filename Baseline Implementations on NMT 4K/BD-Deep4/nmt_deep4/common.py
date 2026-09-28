from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve_path(value: str | os.PathLike[str], base: str | os.PathLike[str] | None = None) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    if base is None:
        return p.resolve()
    return (Path(base) / p).resolve()


def get_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    config_dir = Path(cfg["_config_path"]).parent
    dataset_root = resolve_path(cfg["dataset_root"], config_dir)
    work_dir = resolve_path(cfg.get("work_dir", "./work"), config_dir)
    manifest_dir = work_dir / "manifests"
    sfreq = float(cfg.get("preprocessing", {}).get("sfreq", 100.0))
    sfreq_tag = f"{sfreq:g}".replace(".", "p")
    cache_dir = work_dir / f"cache_{sfreq_tag}hz"
    run_dir = work_dir / "runs" / cfg.get("run_name", "bddeep4")
    return {
        "dataset_root": dataset_root,
        "metadata": dataset_root / "metadata" / "recordings.tsv",
        "work_dir": work_dir,
        "manifest_dir": manifest_dir,
        "manifest": manifest_dir / "records.csv",
        "cache_dir": cache_dir,
        "stats": cache_dir / "train_channel_stats.npz",
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "artifacts_dir": run_dir / "artifacts",
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    for key in ("work_dir", "manifest_dir", "cache_dir", "run_dir", "checkpoint_dir", "artifacts_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def device_from_config(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def stable_seed(*parts: int) -> int:
    # Small deterministic integer mixer; stable across Python sessions.
    x = 0x9E3779B9
    for p in parts:
        x ^= int(p) + 0x9E3779B9 + ((x << 6) & 0xFFFFFFFF) + (x >> 2)
        x &= 0xFFFFFFFF
    return x
