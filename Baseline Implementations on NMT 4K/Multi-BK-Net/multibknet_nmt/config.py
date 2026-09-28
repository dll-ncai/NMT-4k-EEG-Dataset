from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimePaths:
    config_file: Path
    dataset_root: Path
    train_dir: Path
    evaluation_dir: Path
    metadata_tsv: Path
    output_root: Path
    cache_dir: Path
    audit_dir: Path
    runs_dir: Path


def _as_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: str | Path) -> tuple[dict[str, Any], RuntimePaths]:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("The YAML configuration must contain a mapping at its root.")

    for section in (
        "dataset",
        "paths",
        "preprocessing",
        "model",
        "training",
        "evaluation",
    ):
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    dataset_cfg = config["dataset"]
    path_cfg = config["paths"]
    dataset_root = _as_path(dataset_cfg["root"], config_file.parent)
    output_root = _as_path(path_cfg["output_root"], config_file.parent)

    def under_dataset(key: str) -> Path:
        return _as_path(dataset_cfg[key], dataset_root)

    def under_output(key: str) -> Path:
        return _as_path(path_cfg[key], output_root)

    paths = RuntimePaths(
        config_file=config_file,
        dataset_root=dataset_root,
        train_dir=under_dataset("train_folder"),
        evaluation_dir=under_dataset("evaluation_folder"),
        metadata_tsv=under_dataset("metadata_tsv"),
        output_root=output_root,
        cache_dir=under_output("cache_dir"),
        audit_dir=under_output("audit_dir"),
        runs_dir=under_output("runs_dir"),
    )

    channels = dataset_cfg.get("channels", [])
    if not isinstance(channels, list) or not channels:
        raise ValueError("dataset.channels must be a non-empty list.")
    if len(set(channels)) != len(channels):
        raise ValueError("dataset.channels contains duplicates.")

    # Version 1.0.2 default. setdefault keeps existing user configuration intact
    # while making the effective policy explicit in resolved configuration files.
    config["preprocessing"].setdefault(
        "short_recording_policy", "last_complete_window"
    )
    target_sfreq = float(config["preprocessing"]["target_sfreq"])
    window_seconds = float(config["preprocessing"]["window_seconds"])
    crop_start_seconds = float(config["preprocessing"]["crop_start_seconds"])
    max_duration_seconds = float(config["preprocessing"]["max_duration_seconds"])
    short_recording_policy = str(
        config["preprocessing"].get("short_recording_policy", "strict")
    ).lower()
    if crop_start_seconds < 0 or max_duration_seconds < window_seconds:
        raise ValueError(
            "Crop start must be non-negative and max duration must be at least one window."
        )
    if short_recording_policy not in {"strict", "last_complete_window"}:
        raise ValueError(
            "preprocessing.short_recording_policy must be strict or last_complete_window"
        )
    input_samples = round(target_sfreq * window_seconds)
    if input_samples != 6000:
        raise ValueError(
            f"Multi-BK-Net paper configuration requires 6000 input samples; got {input_samples}."
        )

    n_filters = int(config["model"]["total_first_block_filters"])
    kernels = config["model"]["temporal_kernel_samples"]
    if len(kernels) != 5 or n_filters % 5 != 0:
        raise ValueError(
            "The first block requires five kernels and a filter count divisible by five."
        )

    micro_batch = int(config["training"]["micro_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    if micro_batch < 1 or accumulation < 1:
        raise ValueError(
            "Batch size and gradient accumulation must be positive integers."
        )

    return config, paths


def ensure_output_directories(paths: RuntimePaths) -> None:
    for path in (paths.output_root, paths.cache_dir, paths.audit_dir, paths.runs_dir):
        path.mkdir(parents=True, exist_ok=True)


def resolved_config(config: dict[str, Any], paths: RuntimePaths) -> dict[str, Any]:
    result = yaml.safe_load(yaml.safe_dump(config))
    result["resolved_paths"] = {
        "config_file": str(paths.config_file),
        "dataset_root": str(paths.dataset_root),
        "train_dir": str(paths.train_dir),
        "evaluation_dir": str(paths.evaluation_dir),
        "metadata_tsv": str(paths.metadata_tsv),
        "output_root": str(paths.output_root),
        "cache_dir": str(paths.cache_dir),
        "audit_dir": str(paths.audit_dir),
        "runs_dir": str(paths.runs_dir),
    }
    return result
