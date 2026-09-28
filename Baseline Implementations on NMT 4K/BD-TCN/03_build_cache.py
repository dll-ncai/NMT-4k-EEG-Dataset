from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from bdtcn_nmt_utils import (
    cache_path_for_row,
    cache_root,
    edf_path_for_row,
    load_and_preprocess_edf,
    load_config,
    preprocessing_hash,
    read_metadata,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--limit", type=int, default=None, help="Only preprocess the first N metadata rows.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute cache files even if they already exist.",
    )
    return p.parse_args()


def valid_existing_cache(path: Path, n_channels: int) -> bool:
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.ndim == 2 and arr.shape[0] == n_channels and arr.shape[1] > 1
    except Exception:
        return False


def main():
    args = parse_args()
    cfg = load_config(args.config)
    df = read_metadata(cfg)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    hash_id = preprocessing_hash(cfg)
    base = cache_root(cfg, args.config) / hash_id
    base.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing configuration hash: {hash_id}")
    print(f"Cache directory: {base}")
    print(f"Records considered: {len(df):,}")
    print("Existing valid cache files will be skipped.")
    print()

    manifest_rows = []
    n_channels = len(cfg["data"]["channels"])
    skipped = 0
    built = 0
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing EDFs", unit="recording"):
        src = edf_path_for_row(row, cfg)
        dst = cache_path_for_row(row, cfg, args.config)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not args.force and dst.exists() and valid_existing_cache(dst, n_channels):
            skipped += 1
            arr = np.load(dst, mmap_mode="r")
            manifest_rows.append(
                {
                    "file_name": row["file_name"],
                    "split": row["split"],
                    "label": row["label"],
                    "cache_path": str(dst),
                    "n_channels": int(arr.shape[0]),
                    "n_samples": int(arr.shape[1]),
                    "status": "existing",
                }
            )
            continue

        tmp = dst.with_suffix(".tmp.npy")
        try:
            x = load_and_preprocess_edf(src, cfg)
            np.save(tmp, x)
            os.replace(tmp, dst)
            built += 1
            manifest_rows.append(
                {
                    "file_name": row["file_name"],
                    "split": row["split"],
                    "label": row["label"],
                    "cache_path": str(dst),
                    "n_channels": int(x.shape[0]),
                    "n_samples": int(x.shape[1]),
                    "status": "built",
                }
            )
            del x
            gc.collect()
        except Exception as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            failed.append((str(src), repr(e)))
            manifest_rows.append(
                {
                    "file_name": row["file_name"],
                    "split": row["split"],
                    "label": row["label"],
                    "cache_path": str(dst),
                    "n_channels": "",
                    "n_samples": "",
                    "status": f"FAILED: {repr(e)}",
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = base / "cache_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    print()
    print(f"Built   : {built:,}")
    print(f"Skipped : {skipped:,}")
    print(f"Failed  : {len(failed):,}")
    print(f"Manifest: {manifest_path}")

    if failed:
        print("\nFirst failures:")
        for p, e in failed[:20]:
            print(f"  {p}\n    {e}")
        raise RuntimeError(
            f"{len(failed)} recordings failed preprocessing. "
            "Fix them before full training."
        )

    print("\nCache build complete.")


if __name__ == "__main__":
    main()
