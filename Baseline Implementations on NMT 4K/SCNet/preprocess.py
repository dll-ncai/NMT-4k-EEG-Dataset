from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import load_config, resolve_path, save_json
from src.preprocessing import atomic_save_npy, preprocess_edf


REQUIRED_METADATA = {"file_name", "split", "label"}


def resolve_edf(root: Path, file_name: str, split: str, label: str) -> Path:
    name = str(file_name)
    if not name.lower().endswith(".edf"):
        name += ".edf"
    return root / str(split).lower() / str(label).lower() / "edf" / name


def _process_one(args: tuple) -> dict:
    (
        root, cache_dir, file_name, split, label, pp,
    ) = args
    root = Path(root)
    cache_dir = Path(cache_dir)
    edf = resolve_edf(root, file_name, split, label)
    cached_file = f"{file_name}.npy"
    out = cache_dir / cached_file
    target = 1 if str(label).strip().lower() == "abnormal" else 0

    expected_channels = 19 if pp["mode"] == "nmt4k19" else 22
    expected_samples = int(round(float(pp["target_sfreq"]) * float(pp["input_seconds"])))

    if out.exists():
        try:
            arr = np.load(out, mmap_mode="r", allow_pickle=False)
            if arr.shape == (expected_channels, expected_samples):
                return {
                    "file_name": file_name, "split": split, "label": label,
                    "target": target, "cached_file": cached_file, "status": "ok",
                    "message": "resumed-existing", "output_channels": arr.shape[0],
                    "output_samples": arr.shape[1],
                }
        except Exception:
            pass

    if not edf.exists():
        return {
            "file_name": file_name, "split": split, "label": label, "target": target,
            "cached_file": cached_file, "status": "error", "message": f"EDF not found: {edf}",
        }

    try:
        arr, info = preprocess_edf(
            edf_path=edf,
            mode=pp["mode"],
            target_sfreq=float(pp["target_sfreq"]),
            input_seconds=float(pp["input_seconds"]),
            discard_seconds=float(pp["discard_seconds"]),
            bandpass=pp.get("bandpass"),
            clip_uv=pp.get("clip_uv"),
            cache_dtype=pp.get("cache_dtype", "float16"),
        )
        atomic_save_npy(arr, out)
        return {
            "file_name": file_name, "split": split, "label": label, "target": target,
            "cached_file": cached_file, "status": "ok", "message": "processed", **info,
        }
    except Exception as e:
        return {
            "file_name": file_name, "split": split, "label": label, "target": target,
            "cached_file": cached_file, "status": "error",
            "message": f"{type(e).__name__}: {e}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess NMT-4K EDF files for SCNet.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=None, help="Override preprocessing workers.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base = cfg_path.parent
    root = Path(cfg["dataset_root"])
    cache_dir = resolve_path(base, cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    pp = dict(cfg["preprocess"])
    workers = int(args.workers if args.workers is not None else pp.get("workers", 2))

    metadata = root / "metadata" / "recordings.tsv"
    if not metadata.exists():
        raise FileNotFoundError(f"Could not find metadata file: {metadata}")
    df = pd.read_csv(metadata, sep="\t", dtype={"file_name": str})
    df.columns = [str(c).strip() for c in df.columns]
    missing = REQUIRED_METADATA - set(df.columns)
    if missing:
        raise ValueError(f"recordings.tsv is missing columns: {sorted(missing)}")

    signature = {
        "mode": pp["mode"],
        "target_sfreq": float(pp["target_sfreq"]),
        "input_seconds": float(pp["input_seconds"]),
        "discard_seconds": float(pp["discard_seconds"]),
        "bandpass": pp.get("bandpass"),
        "clip_uv": pp.get("clip_uv"),
        "cache_dtype": pp.get("cache_dtype", "float16"),
    }
    meta_path = cache_dir / "cache_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != signature:
            raise RuntimeError(
                "Cache settings changed. Use a new cache_dir (recommended) or delete the old cache "
                "before preprocessing with different settings."
            )
    else:
        save_json(signature, meta_path)

    jobs = [
        (root, cache_dir, str(r.file_name), str(r.split), str(r.label), pp)
        for r in df[["file_name", "split", "label"]].itertuples(index=False)
    ]

    results: list[dict] = []
    if workers <= 1:
        for job in tqdm(jobs, desc="Preprocessing EDF", unit="recording"):
            results.append(_process_one(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing EDF", unit="recording"):
                results.append(fut.result())

    manifest = pd.DataFrame(results)
    # Reorder to metadata order for easy audit.
    order = {name: i for i, name in enumerate(df["file_name"].astype(str))}
    manifest["_order"] = manifest["file_name"].map(order)
    manifest = manifest.sort_values("_order").drop(columns="_order")
    manifest.to_csv(cache_dir / "manifest.csv", index=False)

    ok = int((manifest["status"] == "ok").sum())
    err = int((manifest["status"] != "ok").sum())
    print(f"\nFinished. Cached/resumed: {ok}/{len(manifest)}; errors: {err}")
    if err:
        print(f"Review: {cache_dir / 'manifest.csv'}")
        print(manifest.loc[manifest["status"] != "ok", ["file_name", "message"]].head(20).to_string(index=False))
    else:
        print(f"Manifest: {cache_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
