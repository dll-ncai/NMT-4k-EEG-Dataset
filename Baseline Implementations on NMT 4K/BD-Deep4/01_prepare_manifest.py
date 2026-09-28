from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nmt_deep4.common import ensure_dirs, get_paths, load_config


def resolve_edf(root: Path, split: str, label: str, file_name: str) -> Path:
    split_dir = "train" if split.lower() == "train" else "evaluation"
    label_dir = label.strip().lower()
    name = file_name if file_name.lower().endswith(".edf") else f"{file_name}.edf"
    p = root / split_dir / label_dir / "edf" / name
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ensure_dirs(paths)

    meta = paths["metadata"]
    if not meta.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {meta}")

    df = pd.read_csv(meta, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    required = {"file_name", "split", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"recordings.tsv is missing: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()

    valid_splits = {"train", "evaluation"}
    if not set(df["split"].unique()).issubset(valid_splits):
        raise ValueError(f"Unexpected split values: {sorted(df['split'].unique())}")
    if not set(df["label"].unique()).issubset({"Normal", "Abnormal"}):
        raise ValueError(f"Unexpected labels: {sorted(df['label'].unique())}")

    train_idx = df.index[df["split"] == "train"].to_numpy()
    train_y = (df.loc[train_idx, "label"] == "Abnormal").astype(int).to_numpy()
    val_fraction = float(cfg.get("validation_fraction", 0.15))
    seed = int(cfg.get("seed", 42))
    fit_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_y,
    )

    df["role"] = "evaluation"
    df.loc[fit_idx, "role"] = "train"
    df.loc[val_idx, "role"] = "val"

    edf_paths = []
    cache_paths = []
    missing_files = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Resolving EDF files"):
        p = resolve_edf(paths["dataset_root"], row.split, row.label, str(row.file_name))
        edf_paths.append(str(p))
        cache_paths.append(str(paths["cache_dir"] / str(row.split).lower() / str(row.label).lower() / f"{Path(str(row.file_name)).stem}.npy"))
        if not p.exists():
            missing_files.append(str(p))

    df["edf_path"] = edf_paths
    df["cache_path"] = cache_paths

    if missing_files:
        report = paths["manifest_dir"] / "missing_edf_files.txt"
        report.write_text("\n".join(missing_files), encoding="utf-8")
        raise FileNotFoundError(
            f"{len(missing_files)} EDF files were not found. See: {report}"
        )

    df.to_csv(paths["manifest"], index=False)

    print("\nManifest saved:", paths["manifest"])
    print("\nRole / label counts:")
    print(pd.crosstab(df["role"], df["label"], margins=True))
    print("\nExpected for the released split: train+val=3500, evaluation=1000.")


if __name__ == "__main__":
    main()
