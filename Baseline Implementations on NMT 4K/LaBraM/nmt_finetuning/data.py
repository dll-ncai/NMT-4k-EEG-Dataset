"""Manifest and pickle dataset utilities for preprocessed NMT EEG windows."""

from __future__ import annotations

import csv
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class ManifestEntry:
    path: Path
    split: str
    label: int
    recording_id: str
    window_index: int


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    required = {"file_path", "split", "label", "recording_id", "window_index"}
    entries: list[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                window_index = int(row["window_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label/window index at manifest line {line_number}") from exc
            split = row["split"].strip().lower()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unknown split {split!r} at manifest line {line_number}")
            if label not in {0, 1}:
                raise ValueError(f"Binary label must be 0 or 1 at manifest line {line_number}")
            entries.append(
                ManifestEntry(
                    path=Path(row["file_path"]),
                    split=split,
                    label=label,
                    recording_id=row["recording_id"].strip(),
                    window_index=window_index,
                )
            )
    if not entries:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    return entries


def select_split(
    entries: Iterable[ManifestEntry],
    split: str,
    max_windows_per_recording: int = 0,
) -> list[ManifestEntry]:
    selected = [entry for entry in entries if entry.split == split]
    if max_windows_per_recording <= 0:
        return selected

    grouped: dict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in selected:
        grouped[entry.recording_id].append(entry)

    limited: list[ManifestEntry] = []
    for recording_id in sorted(grouped):
        group = sorted(grouped[recording_id], key=lambda item: item.window_index)
        if len(group) <= max_windows_per_recording:
            limited.extend(group)
            continue
        # Even spacing preserves coverage of the full recording and is deterministic.
        positions = np.linspace(0, len(group) - 1, max_windows_per_recording, dtype=np.int64)
        limited.extend(group[int(position)] for position in positions)
    return limited


class NMTWindowDataset(Dataset):
    """Loads ``{'X': ndarray[21, 2000], 'y': int}`` pickle windows."""

    expected_shape = (21, 2000)

    def __init__(self, entries: Iterable[ManifestEntry], validate_paths: bool = False):
        self.entries = list(entries)
        if not self.entries:
            raise ValueError("Dataset split contains no windows")
        if validate_paths:
            missing = [str(entry.path) for entry in self.entries if not entry.path.is_file()]
            if missing:
                preview = "\n".join(missing[:10])
                raise FileNotFoundError(
                    f"{len(missing)} manifest files do not exist. First missing paths:\n{preview}"
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        entry = self.entries[index]
        try:
            with entry.path.open("rb") as handle:
                sample = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Could not load EEG window: {entry.path}") from exc

        if not isinstance(sample, dict) or "X" not in sample:
            raise ValueError(f"Expected a dictionary containing 'X': {entry.path}")
        array = np.asarray(sample["X"], dtype=np.float32)
        if array.shape != self.expected_shape:
            raise ValueError(
                f"Expected EEG shape {self.expected_shape}, got {array.shape}: {entry.path}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"EEG window contains NaN or Inf: {entry.path}")

        stored_label = int(sample.get("y", entry.label))
        if stored_label != entry.label:
            raise ValueError(
                f"Manifest label {entry.label} != pickle label {stored_label}: {entry.path}"
            )
        signal = torch.from_numpy(np.ascontiguousarray(array))
        label = torch.tensor(float(entry.label), dtype=torch.float32)
        return signal, label, entry.recording_id


def make_recording_balanced_sampler(
    entries: list[ManifestEntry],
    num_samples: int,
    seed: int,
) -> WeightedRandomSampler:
    """Balance labels and recordings while sampling windows with replacement."""
    if num_samples <= 0:
        num_samples = len(entries)

    recording_counts = Counter(entry.recording_id for entry in entries)
    recording_labels: dict[str, int] = {}
    for entry in entries:
        previous = recording_labels.setdefault(entry.recording_id, entry.label)
        if previous != entry.label:
            raise ValueError(f"Recording has conflicting labels: {entry.recording_id}")
    class_recordings = Counter(recording_labels.values())
    if set(class_recordings) != {0, 1}:
        raise ValueError("Training data must contain both binary classes")

    weights = [
        1.0 / (class_recordings[entry.label] * recording_counts[entry.recording_id])
        for entry in entries
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def summarize_entries(entries: Iterable[ManifestEntry]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    entries_list = list(entries)
    for split in ("train", "val", "test"):
        subset = [entry for entry in entries_list if entry.split == split]
        recordings = {(entry.recording_id, entry.label) for entry in subset}
        summary[split] = {
            "windows": len(subset),
            "recordings": len(recordings),
            "normal_recordings": sum(label == 0 for _, label in recordings),
            "abnormal_recordings": sum(label == 1 for _, label in recordings),
        }
    return summary
