"""Audited loading of the released EEGPT checkpoint and experiment checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


KNOWN_PREFIXES = ("module.", "model.", "student.", "backbone.", "network.")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Find tensor weights in common Lightning and plain PyTorch layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint root must be a dictionary.")
    for key in ("state_dict", "model_state_dict", "model", "student"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and any(torch.is_tensor(value) for value in candidate.values()):
            return {str(name): value for name, value in candidate.items() if torch.is_tensor(value)}
    if any(torch.is_tensor(value) for value in checkpoint.values()):
        return {str(name): value for name, value in checkpoint.items() if torch.is_tensor(value)}
    raise KeyError("No tensor state dictionary was found inside the checkpoint.")


def _key_variants(source_key: str) -> list[str]:
    """Generate conservative prefix mappings without broad string replacement."""
    variants = [source_key]
    cleaned = source_key
    changed = True
    while changed:
        changed = False
        for prefix in KNOWN_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                variants.append(cleaned)
                changed = True
    if ".target_encoder." in cleaned:
        variants.append("target_encoder." + cleaned.split(".target_encoder.", 1)[1])
    if cleaned.startswith("encoder."):
        variants.append("target_encoder." + cleaned[len("encoder.") :])
    if not cleaned.startswith("target_encoder."):
        variants.append("target_encoder." + cleaned)
    return list(dict.fromkeys(variants))


def load_pretrained_eegpt(
    model: nn.Module,
    checkpoint_path: str | Path,
    minimum_encoder_coverage: float = 0.95,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load matching weights and fail if too little of EEGPT was actually restored."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"EEGPT checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = _extract_state_dict(checkpoint)
    target = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    used_source_keys: set[str] = set()
    shape_mismatches: list[dict[str, Any]] = []

    for source_key, tensor in source.items():
        base_variants = _key_variants(source_key)
        # The v2 classifier nests the upstream module inside window_encoder;
        # the released checkpoint generally stores target_encoder at its root.
        candidates = base_variants + ["window_encoder." + key for key in base_variants]
        for candidate in list(dict.fromkeys(candidates)):
            if candidate not in target:
                continue
            if tuple(target[candidate].shape) != tuple(tensor.shape):
                shape_mismatches.append(
                    {"source": source_key, "target": candidate, "source_shape": list(tensor.shape), "target_shape": list(target[candidate].shape)}
                )
                continue
            if candidate not in matched:
                matched[candidate] = tensor
                used_source_keys.add(source_key)
                break

    encoder_names = [name for name in target if "target_encoder." in name]
    encoder_elements = sum(target[name].numel() for name in encoder_names)
    loaded_encoder_elements = sum(target[name].numel() for name in encoder_names if name in matched)
    coverage = loaded_encoder_elements / encoder_elements if encoder_elements else 0.0
    missing, unexpected = model.load_state_dict(matched, strict=False)
    report = {
        "checkpoint": str(checkpoint_path),
        "source_tensor_count": len(source),
        "matched_tensor_count": len(matched),
        "encoder_parameter_coverage": float(coverage),
        "missing_model_keys": list(missing),
        "unexpected_model_keys": list(unexpected),
        "unused_source_tensor_count": len(source) - len(used_source_keys),
        "shape_mismatches": shape_mismatches[:100],
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    if coverage < float(minimum_encoder_coverage):
        raise RuntimeError(
            f"Only {coverage:.1%} of EEGPT encoder parameters loaded; required "
            f"{minimum_encoder_coverage:.1%}. Inspect the checkpoint load report."
        )
    print(
        f"Loaded {len(matched)} tensors; EEGPT encoder parameter coverage: {coverage:.2%}."
    )
    return report


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Prevent an interrupted save from corrupting the previous checkpoint."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_experiment_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Restore a v2 experiment model strictly; architecture drift becomes visible."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Not a v2 experiment checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint
