from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd
import torch

from src.common import load_config
from src.preprocessing import SCALP_19, PAPER_21, resolve_channels


def resolve_edf(root: Path, file_name: str, split: str, label: str) -> Path:
    name = str(file_name)
    if not name.lower().endswith(".edf"):
        name += ".edf"
    return root / str(split).lower() / str(label).lower() / "edf" / name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["dataset_root"])
    metadata = root / "metadata" / "recordings.tsv"

    print("=== GPU ===")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("PyTorch CUDA runtime:", torch.version.cuda)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_gb:.2f} GB")

    print("\n=== Dataset ===")
    print("Root:", root)
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    df = pd.read_csv(metadata, sep="\t", dtype={"file_name": str})
    print("Metadata rows:", len(df))
    print("Split counts:", df["split"].value_counts().to_dict())
    print("Label counts:", df["label"].value_counts().to_dict())
    print("Split x label:\n", pd.crosstab(df["split"], df["label"]))

    required = SCALP_19 if cfg["preprocess"]["mode"] == "nmt4k19" else PAPER_21
    print(f"\n=== EDF header spot-check ({cfg['preprocess']['mode']}) ===")
    for _, row in df.head(3).iterrows():
        edf = resolve_edf(root, row["file_name"], row["split"], row["label"])
        print("\n", edf)
        if not edf.exists():
            print("  MISSING")
            continue
        raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
        print("  sfreq:", raw.info["sfreq"], "Hz")
        print("  duration:", raw.n_times / raw.info["sfreq"], "sec")
        print("  channels:", raw.ch_names)
        mapping = resolve_channels(raw, required)
        print("  required-channel mapping OK:", mapping)

    print("\nSetup check finished.")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; fix the PyTorch GPU installation before training.")


if __name__ == "__main__":
    main()
