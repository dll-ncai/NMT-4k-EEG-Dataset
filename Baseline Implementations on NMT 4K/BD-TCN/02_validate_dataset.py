from __future__ import annotations

import argparse
from collections import Counter

import mne
from tqdm import tqdm

from bdtcn_nmt_utils import (
    dataset_root,
    edf_path_for_row,
    find_channel_indices,
    load_config,
    read_metadata,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--check-all-edf",
        action="store_true",
        help="Read every EDF header instead of a representative subset.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    root = dataset_root(cfg)

    print(f"Dataset root: {root}")
    if not root.exists():
        raise FileNotFoundError(root)

    df = read_metadata(cfg)
    print(f"Metadata rows: {len(df):,}")
    print()
    print("Counts by split and label:")
    print(df.groupby(["split", "label"]).size().to_string())
    print()

    missing = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking EDF paths", unit="file"):
        p = edf_path_for_row(row, cfg)
        if not p.exists():
            missing.append(str(p))

    if missing:
        print("\nMissing EDF files:")
        for p in missing[:20]:
            print(" ", p)
        raise FileNotFoundError(f"{len(missing)} EDF files referenced by metadata are missing.")

    print("All EDF paths referenced by recordings.tsv exist.")

    if args.check_all_edf:
        check_df = df
    else:
        # Deterministic representative sample from each split/label.
        parts = []
        for _, group in df.groupby(["split", "label"], sort=True):
            parts.append(group.sort_values("file_name").head(10))
        check_df = __import__("pandas").concat(parts, ignore_index=True)
        print(f"Header-checking representative subset: {len(check_df)} EDFs")
        print("Use --check-all-edf for all recordings.")

    sfreqs = Counter()
    failures = []
    for _, row in tqdm(check_df.iterrows(), total=len(check_df), desc="Checking EDF headers", unit="file"):
        p = edf_path_for_row(row, cfg)
        try:
            raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
            find_channel_indices(raw.ch_names, cfg["data"]["channels"])
            sfreqs[float(raw.info["sfreq"])] += 1
        except Exception as e:
            failures.append((str(p), repr(e)))

    if failures:
        print("\nEDF header/channel failures:")
        for p, e in failures[:20]:
            print(f"  {p}\n    {e}")
        raise RuntimeError(f"{len(failures)} EDF header/channel checks failed.")

    print("Sampling rates seen:")
    for sf, n in sorted(sfreqs.items()):
        print(f"  {sf:g} Hz: {n} checked files")

    expected = {
        ("train", "Normal"): 2796,
        ("train", "Abnormal"): 704,
        ("evaluation", "Normal"): 540,
        ("evaluation", "Abnormal"): 460,
    }
    actual = df.groupby(["split", "label"]).size().to_dict()
    if all(actual.get(k) == v for k, v in expected.items()):
        print("\nNMT-4K-EEG v1.2 expected classification counts match exactly.")
    else:
        print("\nNOTE: counts differ from the NMT-4K-EEG v1.2 paper:")
        for key, value in expected.items():
            print(f"  {key}: expected {value}, found {actual.get(key, 0)}")

    print("\nDataset validation complete.")


if __name__ == "__main__":
    main()
