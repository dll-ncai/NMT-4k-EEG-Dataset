#!/usr/bin/env python
"""Check the CUDA 12.8/PyTorch environment and the bundled LaBraM checkpoint."""

from __future__ import annotations

import argparse
import importlib
import platform
import zipfile
from pathlib import Path

from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "LaBraM/checkpoints/labram-base.pth",
    )
    parser.add_argument("--cpu-ok", action="store_true", help="Do not fail when CUDA is unavailable")
    return parser.parse_args()


def package_version(module_name: str) -> tuple[object | None, str | None, str | None]:
    try:
        module = importlib.import_module(module_name)
        return module, getattr(module, "__version__", "unknown"), None
    except Exception as exc:  # noqa: BLE001 - every import failure belongs in the report
        return None, None, str(exc)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    print(f"Python: {platform.python_version()} ({platform.platform()})")
    if not (Version("3.10") <= Version(platform.python_version()) < Version("3.13")):
        warnings.append("Python 3.10 or 3.11 is recommended for this legacy LaBraM codebase.")

    modules: dict[str, object] = {}
    versions: dict[str, str] = {}
    for name in ("torch", "torchvision", "torchaudio", "numpy", "pandas", "scipy", "sklearn", "timm", "mne"):
        module, version, error = package_version(name)
        if error:
            errors.append(f"Cannot import {name}: {error}")
            continue
        modules[name] = module
        versions[name] = str(version)
        print(f"{name}: {version}")

    torch = modules.get("torch")
    if torch is not None:
        if not versions["torch"].startswith("2.10.0"):
            warnings.append(f"Scripts were prepared for torch 2.10.0; found {versions['torch']}.")
        runtime = str(torch.version.cuda)
        print(f"PyTorch CUDA runtime: {runtime}")
        if not runtime.startswith("12.8"):
            errors.append(f"Expected the cu128 PyTorch build; torch.version.cuda is {runtime}.")
        cuda_available = bool(torch.cuda.is_available())
        print(f"CUDA available: {cuda_available}")
        if not cuda_available and not args.cpu_ok:
            errors.append("CUDA is unavailable. Check the NVIDIA driver and the cu128 PyTorch wheel.")
        if cuda_available:
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            architectures = list(torch.cuda.get_arch_list())
            print(f"GPU: {name}")
            print(f"Compute capability: {capability}")
            print(f"Compiled CUDA architectures: {architectures}")
            if "RTX 50" in name.upper() and capability[0] < 12:
                warnings.append(f"Unexpected compute capability for an RTX 50-series device: {capability}")
            if capability == (12, 0) and "sm_120" not in architectures:
                errors.append(
                    "This PyTorch wheel does not list sm_120 support for the RTX 50-series GPU. "
                    "Reinstall the official torch 2.10.0 cu128 wheel."
                )
            try:
                left = torch.randn(64, 64, device="cuda")
                product = left @ left
                torch.cuda.synchronize()
                print(f"CUDA matrix smoke test: passed ({tuple(product.shape)})")
            except RuntimeError as exc:
                errors.append(f"CUDA matrix smoke test failed: {exc}")

    if {"torch", "torchvision", "torchaudio"}.issubset(versions):
        expected = {"torch": "2.10.0", "torchvision": "0.25.0", "torchaudio": "2.10.0"}
        for name, wanted in expected.items():
            if not versions[name].startswith(wanted):
                errors.append(f"Version trio mismatch: expected {name} {wanted}, found {versions[name]}.")

    if "numpy" in versions:
        numpy_version = Version(versions["numpy"])
        if numpy_version >= Version("2.0"):
            errors.append(
                f"NumPy {numpy_version} is incompatible with several old compiled packages in the supplied "
                "environment. Use NumPy 1.26.4 from requirements_nmt_cuda128.txt."
            )
    if (
        "pandas" in versions
        and "numpy" in versions
        and Version(versions["pandas"]) < Version("2.2.2")
        and Version(versions["numpy"]) >= Version("2.0")
    ):
        errors.append("pandas <2.2.2 is not generally compatible with NumPy 2.x wheels.")
    if "timm" in versions and Version(versions["timm"]) < Version("0.9.16"):
        errors.append(
            f"timm {versions['timm']} is the old upstream LaBraM pin. Use timm 0.9.16 with PyTorch 2.10."
        )

    if not args.checkpoint.is_file():
        errors.append(f"Checkpoint is missing: {args.checkpoint}")
    else:
        print(f"Checkpoint: {args.checkpoint} ({args.checkpoint.stat().st_size:,} bytes)")
        try:
            with zipfile.ZipFile(args.checkpoint) as archive:
                corrupt_member = archive.testzip()
            if corrupt_member is not None:
                errors.append(f"Checkpoint archive has a corrupt member: {corrupt_member}")
        except zipfile.BadZipFile as exc:
            errors.append(f"Checkpoint is truncated or corrupt: {exc}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
