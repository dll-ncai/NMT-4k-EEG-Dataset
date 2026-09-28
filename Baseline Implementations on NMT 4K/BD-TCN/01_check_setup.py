from __future__ import annotations

import platform
import sys

import braindecode
import mne
import torch


def main():
    print("=" * 72)
    print("BD-TCN / NMT-4K Windows environment check")
    print("=" * 72)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"OS              : {platform.platform()}")
    print(f"PyTorch         : {torch.__version__}")
    print(f"Braindecode     : {braindecode.__version__}")
    print(f"MNE             : {mne.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"CUDA runtime    : {torch.version.cuda}")
        print(f"GPU             : {props.name}")
        print(f"VRAM            : {props.total_memory / 1024**3:.2f} GB")
        x = torch.randn(8, 19, 6000, device="cuda")
        print(f"CUDA tensor test: OK {tuple(x.shape)}")
        del x
        torch.cuda.empty_cache()
    else:
        print()
        print("WARNING: CUDA is not available.")
        print("Do not start the full training run until CUDA works.")

    print("=" * 72)


if __name__ == "__main__":
    main()
