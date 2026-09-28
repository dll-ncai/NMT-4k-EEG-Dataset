from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import yaml

from .config import ensure_output_directories, load_config
from .model import build_model, count_trainable_parameters
from .runtime import amp_context, make_grad_scaler
from .utils import atomic_write_json, print_heading


def verify(config_path: str | Path, environment_only: bool = False) -> int:
    config, paths = load_config(config_path)
    ensure_output_directories(paths)
    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "mne": mne.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    print_heading("Environment and CUDA verification")
    for key, value in report.items():
        print(f"{key}: {value}")

    if not torch.cuda.is_available():
        report["status"] = "failed_cuda_unavailable"
        atomic_write_json(paths.audit_dir / "environment_report.json", report)
        print(
            "\nCUDA is unavailable. Reinstall PyTorch from the CUDA 12.8 index and verify the NVIDIA driver."
        )
        return 1

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    report.update(
        {
            "gpu_name": properties.name,
            "gpu_total_memory_gib": float(properties.total_memory / 2**30),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    )
    print(f"gpu_name: {report['gpu_name']}")
    print(f"gpu_total_memory_gib: {report['gpu_total_memory_gib']:.2f}")
    print(f"compute_capability: {report['compute_capability']}")

    # A tiny CUDA kernel catches unsupported RTX 50-series architectures early.
    try:
        smoke = torch.ones(16, device=device)
        if float((smoke * 2).sum().item()) != 32.0:
            raise RuntimeError("unexpected CUDA arithmetic result")
    except Exception as exc:  # noqa: BLE001 - report any CUDA runtime incompatibility
        report["status"] = "failed_cuda_kernel"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(paths.audit_dir / "environment_report.json", report)
        print(f"CUDA kernel test failed: {exc}")
        return 1

    if environment_only:
        report["status"] = "environment_ok"
        atomic_write_json(paths.audit_dir / "environment_report.json", report)
        return 0

    print_heading("Full Multi-BK-Net GPU step")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(config).to(device)
        model.train()
        parameters = count_trainable_parameters(model)
        micro_batch = int(config["training"]["micro_batch_size"])
        channels = len(config["dataset"]["channels"])
        samples = round(
            float(config["preprocessing"]["target_sfreq"])
            * float(config["preprocessing"]["window_seconds"])
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = make_grad_scaler(device, bool(config["training"]["amp"]))
        inputs = torch.randn(micro_batch, channels, samples, device=device)
        targets = torch.arange(micro_batch, device=device) % 2
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, bool(config["training"]["amp"])):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        peak_gib = float(torch.cuda.max_memory_allocated() / 2**30)
        report.update(
            {
                "model_parameters": int(parameters),
                "model_output_shape": list(logits.shape),
                "model_final_feature_samples": int(model.final_feature_samples),
                "verified_micro_batch_size": micro_batch,
                "amp": bool(config["training"]["amp"]),
                "test_loss": float(loss.detach().cpu()),
                "peak_allocated_memory_gib": peak_gib,
                "status": "ok",
            }
        )
        print(f"trainable_parameters: {parameters:,}")
        print(f"output_shape: {tuple(logits.shape)}")
        print(f"final_feature_samples: {model.final_feature_samples}")
        print(f"peak_allocated_memory_gib: {peak_gib:.2f}")
        print("Full forward/backward test: PASSED")
    except torch.cuda.OutOfMemoryError as exc:
        report["status"] = "failed_out_of_memory"
        report["error"] = str(exc)
        atomic_write_json(paths.audit_dir / "environment_report.json", report)
        print(
            "GPU out of memory. Set micro_batch_size: 8 and "
            "gradient_accumulation_steps: 8 in config.yaml, then rerun."
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - return a complete diagnostic report
        report["status"] = "failed_model_step"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(paths.audit_dir / "environment_report.json", report)
        print(f"Model verification failed: {type(exc).__name__}: {exc}")
        return 1

    atomic_write_json(paths.audit_dir / "environment_report.json", report)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify CUDA and Multi-BK-Net execution."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--environment-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(verify(args.config, args.environment_only))


if __name__ == "__main__":
    main()
