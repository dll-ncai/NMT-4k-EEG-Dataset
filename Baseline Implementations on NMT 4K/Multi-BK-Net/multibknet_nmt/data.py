from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def compute_window_starts(
    n_samples: int,
    window_size: int,
    stride: int,
    include_final_overlapping_window: bool = True,
) -> list[int]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if n_samples < window_size:
        return []
    starts = list(range(0, n_samples - window_size + 1, stride))
    final_start = n_samples - window_size
    if include_final_overlapping_window and starts[-1] != final_start:
        starts.append(final_start)
    return starts


@dataclass(frozen=True)
class WindowEntry:
    recording_row: int
    start: int


class WindowedRecordingDataset(Dataset):
    def __init__(
        self,
        recording_manifest: pd.DataFrame,
        window_size: int,
        stride: int,
        include_final_overlapping_window: bool = True,
    ) -> None:
        required = {"recording_id", "label", "cache_path", "n_samples"}
        missing = required - set(recording_manifest.columns)
        if missing:
            raise ValueError(
                f"Processed manifest is missing columns: {sorted(missing)}"
            )

        self.recordings = recording_manifest.reset_index(drop=True).copy()
        self.cache_paths = tuple(
            Path(value) for value in self.recordings["cache_path"].astype(str)
        )
        self.recording_ids = tuple(
            self.recordings["recording_id"].astype(str).tolist()
        )
        self.recording_labels = tuple(
            self.recordings["label"].astype(int).tolist()
        )
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.include_final = bool(include_final_overlapping_window)
        self.memmap_cache_size = int(
            os.environ.get("MBK_RUNTIME_MEMMAP_CACHE_SIZE", "128")
        )
        if self.memmap_cache_size < 0:
            raise ValueError("MBK_RUNTIME_MEMMAP_CACHE_SIZE cannot be negative")
        self._array_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.entries: list[WindowEntry] = []
        labels: list[int] = []

        for row_index, row in self.recordings.iterrows():
            starts = compute_window_starts(
                int(row["n_samples"]), self.window_size, self.stride, self.include_final
            )
            if not starts:
                raise ValueError(
                    f"Recording {row['recording_id']} contains no complete windows."
                )
            self.entries.extend(WindowEntry(row_index, start) for start in starts)
            labels.extend([int(row["label"])] * len(starts))
        self.window_labels = np.asarray(labels, dtype=np.int64)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Each Windows DataLoader worker opens its own memory maps. Never pickle
        # live mappings or file handles from the parent process.
        state["_array_cache"] = OrderedDict()
        return state

    def _cached_array(self, path: Path) -> np.ndarray:
        key = str(path)
        cached = self._array_cache.pop(key, None)
        if cached is not None:
            self._array_cache[key] = cached
            return cached

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.memmap_cache_size == 0:
            return array

        self._array_cache[key] = array
        while len(self._array_cache) > self.memmap_cache_size:
            _old_key, old_array = self._array_cache.popitem(last=False)
            mapping = getattr(old_array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        return array

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        entry = self.entries[index]
        recording_row = entry.recording_row
        array = self._cached_array(self.cache_paths[recording_row])
        stop = entry.start + self.window_size
        window = np.array(array[:, entry.start : stop], dtype=np.float32, copy=True)
        if window.shape[1] != self.window_size:
            raise RuntimeError(
                f"Cached recording {self.recording_ids[recording_row]} returned "
                f"a short window {window.shape}."
            )
        return (
            torch.from_numpy(window),
            torch.tensor(self.recording_labels[recording_row], dtype=torch.long),
            self.recording_ids[recording_row],
        )


def window_parameters(config: dict) -> tuple[int, int, bool]:
    preprocessing = config["preprocessing"]
    sfreq = float(preprocessing["target_sfreq"])
    window_size = round(sfreq * float(preprocessing["window_seconds"]))
    stride = round(sfreq * float(preprocessing["window_stride_seconds"]))
    include_final = bool(preprocessing.get("include_final_overlapping_window", True))
    return window_size, stride, include_final
