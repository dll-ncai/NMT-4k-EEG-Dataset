from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import stable_seed


SCALP_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]


def label_to_int(label: str) -> int:
    label = str(label).strip().lower()
    if label == "normal":
        return 0
    if label == "abnormal":
        return 1
    raise ValueError(f"Unknown label: {label!r}")


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"file_name", "split", "label", "role", "edf_path", "cache_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return df


def load_channel_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    mean = z["mean"].astype(np.float32)
    std = z["std"].astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean[:, None], std[:, None]


def _window_from_array(
    arr: np.ndarray,
    start: int,
    n_times: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Extract, pad if needed, sanitize and z-normalize one EEG window."""
    if arr.ndim != 2:
        raise ValueError(f"Expected [channels,time] cache array, got {arr.shape}")
    n = int(arr.shape[1])
    if n >= n_times:
        x = np.asarray(arr[:, start:start + n_times], dtype=np.float32)
    else:
        x = np.zeros((arr.shape[0], n_times), dtype=np.float32)
        x[:, :n] = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - mean) / std
    return np.ascontiguousarray(x, dtype=np.float32)


class RandomWindowDataset(Dataset):
    """One or more deterministic-random windows from each training recording per epoch.

    The window location is a pure function of (seed, epoch, recording index,
    repetition index). This makes training data order reproducible and makes
    mid-epoch checkpoint resume practical even on Windows DataLoader workers.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        n_times: int,
        windows_per_recording: int,
        mean: np.ndarray,
        std: np.ndarray,
        seed: int = 42,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.n_times = int(n_times)
        self.windows_per_recording = int(windows_per_recording)
        self.mean = mean
        self.std = std
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_recording

    def __getitem__(self, idx: int):
        rec_idx = idx // self.windows_per_recording
        rep_idx = idx % self.windows_per_recording
        row = self.records.iloc[rec_idx]
        arr = np.load(row.cache_path, mmap_mode="r")
        max_start = max(0, int(arr.shape[1]) - self.n_times)
        if max_start > 0:
            rng = np.random.default_rng(stable_seed(self.seed, self.epoch, rec_idx, rep_idx))
            start = int(rng.integers(0, max_start + 1))
        else:
            start = 0
        x = _window_from_array(arr, start, self.n_times, self.mean, self.std)
        y = np.float32(label_to_int(row.label))
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


@dataclass(frozen=True)
class WindowRef:
    record_index: int
    start: int


class FixedWindowDataset(Dataset):
    """Deterministic windows for validation/evaluation.

    If windows_per_recording is None, sliding windows with stride_samples are
    used. Otherwise, a fixed number of uniformly spaced windows is used.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        n_times: int,
        mean: np.ndarray,
        std: np.ndarray,
        windows_per_recording: int | None = None,
        stride_samples: int | None = None,
        max_windows_per_recording: int | None = None,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.n_times = int(n_times)
        self.mean = mean
        self.std = std
        self.refs: list[WindowRef] = []

        for rec_idx, row in self.records.iterrows():
            arr = np.load(row.cache_path, mmap_mode="r")
            n = int(arr.shape[1])
            max_start = max(0, n - self.n_times)
            if windows_per_recording is not None:
                k = max(1, int(windows_per_recording))
                starts = np.linspace(0, max_start, num=k, dtype=np.int64) if max_start else np.array([0])
            else:
                stride = int(stride_samples or self.n_times)
                starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
                if starts.size == 0:
                    starts = np.array([0], dtype=np.int64)
                # Include the end of the recording even when stride does not land on it.
                if starts[-1] != max_start and max_start > 0:
                    starts = np.concatenate([starts, np.array([max_start], dtype=np.int64)])
                if max_windows_per_recording is not None and len(starts) > max_windows_per_recording:
                    sel = np.linspace(0, len(starts) - 1, num=max_windows_per_recording, dtype=np.int64)
                    starts = starts[sel]
            for s in np.unique(starts):
                self.refs.append(WindowRef(rec_idx, int(s)))

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int):
        ref = self.refs[idx]
        row = self.records.iloc[ref.record_index]
        arr = np.load(row.cache_path, mmap_mode="r")
        x = _window_from_array(arr, ref.start, self.n_times, self.mean, self.std)
        y = np.float32(label_to_int(row.label))
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(ref.record_index, dtype=torch.long),
        )
