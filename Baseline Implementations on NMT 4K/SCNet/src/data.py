from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class CachedEEGDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, cache_dir: str | Path) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        path = self.cache_dir / str(row["cached_file"])
        x = np.load(path, mmap_mode=None, allow_pickle=False)
        # Cached float16 saves ~50% disk; compute starts from float32 and AMP can
        # cast convolution ops to float16 on the GPU.
        x = np.asarray(x, dtype=np.float32)
        y = int(row["target"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), str(row["file_name"])


def make_or_load_splits(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = output_dir / "split_assignments.csv"

    good = manifest[manifest["status"] == "ok"].copy()
    train_pool = good[good["split"].str.lower() == "train"].copy()
    evaluation = good[good["split"].str.lower() == "evaluation"].copy()

    if split_file.exists():
        assignments = pd.read_csv(split_file, dtype={"file_name": str})
        map_set = dict(zip(assignments["file_name"], assignments["model_split"]))
        train_pool["model_split"] = train_pool["file_name"].map(map_set)
        missing = train_pool["model_split"].isna()
        if missing.any():
            raise RuntimeError(
                "Existing split_assignments.csv does not cover the current cache. "
                "Delete it only if you intentionally want to regenerate the validation split."
            )
        train_df = train_pool[train_pool["model_split"] == "train"].drop(columns="model_split")
        val_df = train_pool[train_pool["model_split"] == "val"].drop(columns="model_split")
        return train_df, val_df, evaluation

    idx_train, idx_val = train_test_split(
        np.arange(len(train_pool)),
        test_size=val_fraction,
        random_state=seed,
        stratify=train_pool["target"].to_numpy(),
    )
    train_df = train_pool.iloc[idx_train].copy()
    val_df = train_pool.iloc[idx_val].copy()

    assignments = pd.concat([
        train_df[["file_name"]].assign(model_split="train"),
        val_df[["file_name"]].assign(model_split="val"),
    ], ignore_index=True)
    assignments.to_csv(split_file, index=False)
    return train_df, val_df, evaluation
