from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config, project_path
from src.preprocess_utils import preprocess_edf


def valid_cache(path: Path, expected_shape: tuple[int, int]) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return tuple(arr.shape) == expected_shape and arr.dtype == np.float32
    except Exception:
        return False


def atomic_save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="Recompute even if valid cache files already exist.")
    args = ap.parse_args()
    cfg = load_config(args.config)

    manifest_path = project_path(cfg, cfg["data"]["manifest_file"])
    manifest = pd.read_csv(manifest_path)
    cache_dir = Path(cfg["data"]["cache_dir"])
    base_dir = cache_dir / "base"
    rev_dir = cache_dir / "reverse"
    base_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    target_n = int(round(float(data_cfg["target_sfreq_hz"]) * float(data_cfg["segment_seconds"])))
    expected_shape = (target_n, int(cfg["model"]["input_channels"]))
    require_full_second = bool(data_cfg.get("require_full_second_segment", True))
    use_aug = bool(data_cfg.get("time_reverse_augmentation", True))

    status_rows = []
    failed_log = cache_dir / "preprocess_failures.csv"

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Preprocessing EDFs"):
        rid = str(row.record_id)
        role = str(row.role).lower()
        base_path = base_dir / f"{rid}.npy"
        rev_path = rev_dir / f"{rid}.npy"
        need_second = use_aug and role == "fit"

        base_ok = (not args.force) and valid_cache(base_path, expected_shape)
        rev_needed_now = need_second and not ((not args.force) and valid_cache(rev_path, expected_shape))

        if base_ok and not rev_needed_now:
            status_rows.append({"record_id": rid, "status": "cached", "base": str(base_path), "reverse": str(rev_path) if rev_path.exists() else ""})
            continue

        try:
            result = preprocess_edf(
                edf_path=row.edf_path,
                filter_low_hz=float(data_cfg["filter_low_hz"]),
                filter_high_hz=float(data_cfg["filter_high_hz"]),
                filter_order=int(data_cfg["filter_order"]),
                target_sfreq_hz=float(data_cfg["target_sfreq_hz"]),
                segment_seconds=float(data_cfg["segment_seconds"]),
                normalization=str(data_cfg["normalization"]),
                need_second_segment=need_second,
            )
            if args.force or not base_ok:
                atomic_save_npy(base_path, result["first"])

            reverse_saved = False
            if need_second and result["second_reversed"] is not None:
                enough = result["second_real_samples"] >= target_n if require_full_second else result["second_real_samples"] > 0
                if enough:
                    atomic_save_npy(rev_path, result["second_reversed"])
                    reverse_saved = True
                elif rev_path.exists() and args.force:
                    rev_path.unlink()

            status_rows.append({
                "record_id": rid,
                "status": "ok",
                "base": str(base_path),
                "reverse": str(rev_path) if reverse_saved or rev_path.exists() else "",
                "first_real_samples": result["first_real_samples"],
                "second_real_samples": result["second_real_samples"],
                "orig_sfreq": result["orig_sfreq"],
            })
        except Exception as e:
            with failed_log.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["record_id", "edf_path", "error"])
                writer.writerow([rid, row.edf_path, repr(e)])
            status_rows.append({"record_id": rid, "status": "failed", "error": repr(e)})

    status = pd.DataFrame(status_rows)
    status.to_csv(cache_dir / "cache_index.csv", index=False)
    failures = int((status["status"] == "failed").sum()) if not status.empty else 0
    print(f"\nCache directory: {cache_dir}")
    print(f"Base cached: {len(list(base_dir.glob('*.npy')))}")
    print(f"Reverse-augmentation cached: {len(list(rev_dir.glob('*.npy')))}")
    print(f"Failures this run: {failures}")
    if failures:
        print(f"See: {failed_log}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
