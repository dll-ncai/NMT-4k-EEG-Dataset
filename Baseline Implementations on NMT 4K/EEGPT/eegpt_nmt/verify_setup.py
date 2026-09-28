"""Check dependencies, CUDA, metadata, checkpoint compatibility, and one forward pass."""

from __future__ import annotations

import argparse

import mne
import pandas as pd
import torch
from packaging.version import Version

from .checkpoint import load_pretrained_eegpt
from .config import load_config, resolve_path
from .model import EEGPTMILClassifier
from .train import model_options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    metadata_path = resolve_path(config, config["paths"]["metadata_csv"])
    checkpoint_path = resolve_path(config, config["paths"]["pretrained_checkpoint"])
    assert metadata_path is not None and checkpoint_path is not None
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    print(f"Metadata rows: {len(pd.read_csv(metadata_path))}")
    print(f"MNE: {mne.__version__}")
    if Version(mne.__version__) < Version("1.9.0"):
        raise RuntimeError(
            "MNE 1.9.0 or newer is required for tolerant parsing of optional NMT EDF patient fields. "
            "Run: python -m pip install --upgrade \"mne==1.9.0\""
        )
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = EEGPTMILClassifier(**model_options(config))
    load_pretrained_eegpt(model, checkpoint_path, config["model"]["minimum_encoder_coverage"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dummy_windows = torch.zeros((1, 2, 21, 1024), device=device)
    dummy_mask = torch.ones((1, 2), dtype=torch.bool, device=device)
    with torch.no_grad():
        output = model(dummy_windows, dummy_mask, encode_chunk_size=2)
    print(f"Forward-pass recording logit shape: {tuple(output['recording_logits'].shape)}")
    print("Setup verification passed.")


if __name__ == "__main__":
    main()
