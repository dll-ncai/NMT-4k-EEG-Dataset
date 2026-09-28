from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class CachedEEGDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: str | Path,
        role: str,
        include_reverse_augmentation: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.samples: List[Tuple[Path, int, str, str]] = []
        subset = manifest.loc[manifest["role"].str.lower() == role.lower()].copy()

        for row in subset.itertuples(index=False):
            record_id = str(row.record_id)
            y = int(row.y)
            label = str(row.label)
            base = self.cache_dir / "base" / f"{record_id}.npy"
            if not base.exists():
                raise FileNotFoundError(f"Missing cache: {base}. Run 02_preprocess.py first.")
            self.samples.append((base, y, record_id, label))

            if include_reverse_augmentation:
                rev = self.cache_dir / "reverse" / f"{record_id}.npy"
                if rev.exists():
                    self.samples.append((rev, y, record_id + "__rev", label))

        if not self.samples:
            raise RuntimeError(f"No samples found for role={role}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y, sample_id, label = self.samples[idx]
        x = np.load(path, allow_pickle=False)
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), sample_id, label

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]


def make_weighted_sampler(labels: List[int], seed: int = 42) -> WeightedRandomSampler:
    labels_np = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_np, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Both classes must exist. Counts={counts.tolist()}")
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels_np]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
