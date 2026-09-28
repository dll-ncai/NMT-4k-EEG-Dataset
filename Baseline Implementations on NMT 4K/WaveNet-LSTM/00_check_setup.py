from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import mne
import pandas as pd
import torch

from src.config import load_config, dataset_path
from src.model import build_model
from src.preprocess_utils import resolve_scalp_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model-forward", action="store_true", help="Run a full 60 s synthetic forward pass on GPU.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["dataset_root"])
    metadata = dataset_path(cfg, cfg["data"]["metadata_file"])

    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    print(f"Dataset root exists: {root.exists()} -> {root}")
    print(f"Metadata exists: {metadata.exists()} -> {metadata}")
    if not metadata.exists():
        raise SystemExit("recordings.tsv was not found. Fix data.dataset_root or data.metadata_file in config.yaml")

    df = pd.read_csv(metadata, sep="\t")
    print(f"Metadata rows: {len(df)}")
    print(f"Metadata columns: {list(df.columns)}")

    # Find one EDF from metadata using the expected released directory structure.
    normalized = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    fn_col, split_col, label_col = normalized["file_name"], normalized["split"], normalized["label"]
    row = df.iloc[0]
    rid = str(row[fn_col])
    if not rid.lower().endswith(".edf"):
        rid_edf = rid + ".edf"
    else:
        rid_edf = rid
    edf = root / str(row[split_col]).lower() / str(row[label_col]).lower() / "edf" / rid_edf
    print(f"Example EDF: {edf}")
    if edf.exists():
        raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
        picks, _ = resolve_scalp_channels(raw)
        print(f"Example sfreq: {raw.info['sfreq']} Hz")
        print(f"Example duration: {raw.times[-1]:.2f} s")
        print(f"Resolved 19 scalp channels: {picks}")
        raw.close()
    else:
        print("WARNING: example EDF not found at expected path. 01_make_splits.py will report all missing paths.")

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.model_forward:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available; not running full forward test.")
        device = torch.device("cuda")
        model = model.to(device).eval()
        samples = int(cfg["data"]["target_sfreq_hz"] * cfg["data"]["segment_seconds"])
        channels = int(cfg["model"]["input_channels"])
        x = torch.zeros((1, samples, channels), device=device, dtype=torch.float32)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            y = model(x)
        print(f"Synthetic forward output shape: {tuple(y.shape)}")
        print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
