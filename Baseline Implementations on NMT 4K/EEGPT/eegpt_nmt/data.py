"""Recording-balanced data loading for multiple-instance learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def resolve_tensor_path(value: str, project_root: Path) -> Path:
    """Interpret portable POSIX-style manifest paths on Windows or Linux."""
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_recording_tensor(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the new dictionary format while giving a clear old-format error."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "windows" not in payload:
        raise ValueError(
            f"{path} uses the old window tensor format. Re-run the v2 preprocessing script."
        )
    windows = payload["windows"]
    starts = payload.get("window_start_seconds", torch.arange(len(windows), dtype=torch.float32) * 4.0)
    if windows.ndim != 3 or tuple(windows.shape[1:]) != (21, 1024):
        raise ValueError(f"Expected [windows, 21, 1024] in {path}, found {tuple(windows.shape)}")
    return windows.to(torch.float32), starts.to(torch.float32)


class RecordingBagDataset(Dataset):
    """Return a balanced bag of windows and one label for each recording.

    ``windows_per_recording=0`` returns all windows and therefore requires a
    DataLoader batch size of one. Training uses a fixed positive number.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        project_root: str | Path,
        split: str,
        windows_per_recording: int,
        training: bool,
        seed: int,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        manifest = pd.read_csv(manifest_csv)
        self.records = manifest[manifest["experiment_split"] == split].reset_index(drop=True)
        if self.records.empty:
            raise ValueError(f"No recordings found for experiment_split={split!r}")
        self.windows_per_recording = int(windows_per_recording)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Change random training windows deterministically at every epoch."""
        self.epoch = int(epoch)

    def _choose_indices(self, number_of_windows: int, index: int) -> np.ndarray:
        if self.windows_per_recording == 0 or number_of_windows <= self.windows_per_recording:
            return np.arange(number_of_windows, dtype=np.int64)
        if self.training:
            generator = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            return np.sort(generator.choice(number_of_windows, self.windows_per_recording, replace=False))
        # Validation uses fixed uniformly spaced windows so every epoch is comparable.
        return np.linspace(0, number_of_windows - 1, self.windows_per_recording, dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        tensor_path = resolve_tensor_path(row["tensor_path"], self.project_root)
        windows, starts = load_recording_tensor(tensor_path)
        chosen = self._choose_indices(len(windows), index)
        selected_windows = windows[torch.as_tensor(chosen, dtype=torch.long)]
        selected_starts = starts[torch.as_tensor(chosen, dtype=torch.long)]

        # Fixed-size bags allow batching. Short recordings are zero padded, and
        # the mask makes the model mathematically ignore the padding.
        target_size = self.windows_per_recording
        if target_size > 0 and len(selected_windows) < target_size:
            padded = torch.zeros((target_size, *selected_windows.shape[1:]), dtype=selected_windows.dtype)
            padded_starts = torch.full((target_size,), -1.0, dtype=selected_starts.dtype)
            mask = torch.zeros(target_size, dtype=torch.bool)
            padded[: len(selected_windows)] = selected_windows
            padded_starts[: len(selected_starts)] = selected_starts
            mask[: len(selected_windows)] = True
            selected_windows, selected_starts = padded, padded_starts
        else:
            mask = torch.ones(len(selected_windows), dtype=torch.bool)

        return {
            "windows": selected_windows,
            "mask": mask,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "recording_id": str(row["recording_id"]),
            "window_start_seconds": selected_starts,
        }


def recording_class_counts(dataset: RecordingBagDataset) -> tuple[int, int]:
    """Return negative/positive recording counts for BCE class weighting."""
    labels = dataset.records["label"].astype(int)
    negatives = int((labels == 0).sum())
    positives = int((labels == 1).sum())
    if negatives == 0 or positives == 0:
        raise ValueError("The internal training split must contain both classes.")
    return negatives, positives

