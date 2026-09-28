from __future__ import annotations

import platform
import sys

import torch


def main():
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA runtime:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / (1024**3):.2f} GB")
        x = torch.randn(256, 256, device="cuda")
        y = x @ x
        print("CUDA smoke test: PASS", tuple(y.shape))
    else:
        print("CUDA smoke test: FAIL (fix the CUDA/PyTorch installation before the full run)")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
