#!/usr/bin/env python3
"""Preprocess NMT EDF recordings and build verified window metadata in one run.

This script is designed for the NMT -> LaBraM fine-tuning workflow.  It:

1. resolves EDF files listed in data/metadata.csv;
2. selects the canonical 21 EEG channels in a fixed order;
3. band-pass filters, applies a mains notch, and resamples to 200 Hz;
4. writes non-overlapping 10-second ``{"X": ..., "y": ...}`` pickle windows;
5. writes every pickle through a temporary file and atomically publishes it;
6. immediately reads the published bytes back, checks their SHA-256 digest,
   unpickles them, and validates keys, shape, label, dtype, and finite values;
7. records the source EDF and time range for every verified window; and
8. creates ``windows_inspection_report.csv`` plus an EDF-level error report.

The per-recording JSONL state makes an interrupted run safely resumable.  A
window is only entered in that state after the final pickle has passed the
read-back validation.  Rerun with --resume after an interruption.

The CSV begins with the same columns used by the project's older
``pkl_csv.py`` report (file_path, status, shape_x, label, error), so the
fine-tuning manifest builder can consume it without a separate verification
scan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import re
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import mne
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


SCRIPT_VERSION = "1.0.0"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "preprocess" else SCRIPT_DIR

TARGET_CHANNELS: Tuple[str, ...] = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ",
)

CHANNEL_ALIASES = {
    "M1": "A1",
    "M2": "A2",
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

WINDOW_FIELDS: Tuple[str, ...] = (
    # Compatibility columns expected by the existing manifest preparation code.
    "file_path",
    "status",
    "shape_x",
    "label",
    "error",
    # Provenance and window lineage.
    "split",
    "source_split",
    "label_name",
    "recording_id",
    "patient_id",
    "window_index",
    "start_sample",
    "end_sample",
    "start_sec",
    "end_sec",
    "source_edf_path",
    "source_edf_name",
    # Reproducibility and integrity fields.
    "sampling_rate_hz",
    "original_sampling_rate_hz",
    "channels",
    "dtype",
    "file_size_bytes",
    "sha256",
    "validation",
    "pipeline_fingerprint",
    "source_signature",
    "created_at_utc",
)

EDF_FIELDS: Tuple[str, ...] = (
    "source_row_number",
    "source_edf_path",
    "source_edf_name",
    "metadata_file_path",
    "recording_id",
    "patient_id",
    "split",
    "source_split",
    "label",
    "label_name",
    "status",
    "error",
    "original_sampling_rate_hz",
    "target_sampling_rate_hz",
    "original_channel_count",
    "selected_channel_count",
    "selected_channels",
    "duration_sec_after_resampling",
    "samples_after_resampling",
    "windows_expected",
    "windows_written_this_run",
    "windows_reused",
    "discarded_tail_samples",
    "effective_high_hz",
    "notch_applied_hz",
    "processing_seconds",
    "source_signature",
    "pipeline_fingerprint",
    "completed_at_utc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_channel_name(name: str) -> str:
    """Convert common EDF channel spellings to the canonical NMT names."""
    value = str(name).strip().upper()
    value = re.sub(r"^EEG[\s:_-]*", "", value)
    value = re.sub(r"[\s:_-]*(REF|AVG|LE|AR)$", "", value)
    value = value.replace(".", "").strip()
    return CHANNEL_ALIASES.get(value, value)


def safe_recording_id(value: str) -> str:
    """Return a filename-safe recording id without permitting path traversal."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError(f"Invalid/empty recording_id: {value!r}")
    return cleaned


def normalise_split(value: Any) -> Tuple[str, str]:
    source = str(value or "").strip().lower()
    if source in {"train", "training"}:
        return "train", source or "train"
    if source in {"evaluation", "eval", "test", "testing"}:
        return "eval", source
    raise ValueError(f"Unsupported split {value!r}; expected train or evaluation/eval/test")


def parse_label(row: Mapping[str, Any]) -> Tuple[int, str]:
    numeric = str(row.get("label", "")).strip()
    if numeric:
        try:
            number = float(numeric)
            if number in (0.0, 1.0):
                label = int(number)
                return label, "abnormal" if label == 1 else "normal"
        except ValueError:
            pass

    text = str(row.get("Label (Normal/Abnormal)", "")).strip().lower()
    if "abnormal" in text:
        return 1, "abnormal"
    if text == "normal" or text.startswith("normal "):
        return 0, "normal"
    raise ValueError(
        "Could not parse binary label from columns 'label' or "
        f"'Label (Normal/Abnormal)': {numeric!r}, {text!r}"
    )


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes in the destination directory and publish with os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".nmt_tmp_{path.name}.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".nmt_tmp_{path.name}.{os.getpid()}.{uuid.uuid4().hex}"
    count = 0
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return count
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one durable line; each recording has its own file, so no lock is needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_latest_jsonl_rows(path: Path) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """Load the newest valid row for every window index; ignore torn JSONL lines."""
    rows: Dict[int, Dict[str, Any]] = {}
    malformed = 0
    if not path.exists():
        return rows, malformed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows[int(row["window_index"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                malformed += 1
    return rows, malformed


def clean_stale_temporary_files(output_root: Path) -> int:
    """Remove only temporary files created by this script."""
    removed = 0
    if not output_root.exists():
        return removed
    for split in ("train", "eval"):
        folder = output_root / split
        if not folder.exists():
            continue
        for path in folder.glob(".nmt_tmp_*"):
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def build_edf_index(raw_root: Path) -> Dict[str, List[Path]]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw EDF root does not exist: {raw_root}")
    index: Dict[str, List[Path]] = defaultdict(list)
    for base, _, files in os.walk(raw_root):
        for filename in files:
            if filename.lower().endswith(".edf"):
                index[filename.lower()].append(Path(base) / filename)
    return dict(index)


def relative_path_after_dataset_marker(path_text: str) -> Optional[Path]:
    normalised = str(path_text).replace("\\", "/")
    marker = "nmt-4k-eeg/"
    position = normalised.lower().find(marker)
    if position < 0:
        return None
    tail = normalised[position + len(marker):]
    return Path(*[part for part in tail.split("/") if part])


def resolve_edf_path(
    metadata_path: str,
    filename: str,
    raw_root: Path,
    output_split: str,
    label_name: str,
    index: Mapping[str, List[Path]],
) -> Path:
    candidates: List[Path] = []
    if metadata_path:
        candidates.append(Path(metadata_path))
        relative = relative_path_after_dataset_marker(metadata_path)
        if relative is not None:
            candidates.append(raw_root / relative)

    source_split = "train" if output_split == "train" else "evaluation"
    candidates.extend(
        (
            raw_root / source_split / label_name / "edf" / filename,
            raw_root / source_split / label_name / filename,
            raw_root / output_split / label_name / "edf" / filename,
            raw_root / output_split / label_name / filename,
        )
    )

    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    matches = index.get(filename.lower(), [])
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"EDF not found under {raw_root}: {filename}")
    raise FileNotFoundError(
        f"EDF filename is ambiguous ({len(matches)} matches): {filename}; "
        "correct file_path in metadata.csv"
    )


def load_metadata_tasks(
    metadata_csv: Path,
    raw_root: Path,
    output_root: Path,
    state_root: Path,
    edf_index: Mapping[str, List[Path]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv}")

    tasks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    semantic_rows: List[Dict[str, Any]] = []
    safe_ids: Dict[str, str] = {}

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Metadata CSV has no header: {metadata_csv}")

        for row_number, source_row in enumerate(reader, start=2):
            row = {str(k): ("" if v is None else v) for k, v in source_row.items()}
            metadata_file_path = str(row.get("file_path", "")).strip()
            filename = str(row.get("File Name", "")).strip()
            if not filename and metadata_file_path:
                filename = Path(metadata_file_path.replace("\\", "/")).name

            raw_recording_id = str(row.get("recording_id", "")).strip()
            if not raw_recording_id and filename:
                raw_recording_id = Path(filename).stem
            patient_id = str(row.get("patient_id", "")).strip()

            base: Dict[str, Any] = {
                "source_row_number": row_number,
                "metadata_file_path": metadata_file_path,
                "source_edf_name": filename,
                "recording_id": raw_recording_id,
                "patient_id": patient_id,
            }

            try:
                if not filename:
                    raise ValueError("Neither 'File Name' nor 'file_path' supplies an EDF filename")
                split, source_split = normalise_split(row.get("split", ""))
                label, label_name = parse_label(row)
                safe_id = safe_recording_id(raw_recording_id)

                collision_key = safe_id.lower()
                if collision_key in safe_ids:
                    raise ValueError(
                        f"recording_id filename collision: {raw_recording_id!r} and "
                        f"{safe_ids[collision_key]!r} both map to {safe_id!r}"
                    )
                safe_ids[collision_key] = raw_recording_id

                source_path = resolve_edf_path(
                    metadata_file_path,
                    filename,
                    raw_root,
                    split,
                    label_name,
                    edf_index,
                )
                state_key = f"{split}/{safe_id}"
                task = {
                    **base,
                    "source_edf_path": str(source_path),
                    "split": split,
                    "source_split": source_split,
                    "label": label,
                    "label_name": label_name,
                    "safe_recording_id": safe_id,
                    "output_dir": str(output_root / split),
                    "window_state_path": str(state_root / "windows" / split / f"{safe_id}.jsonl"),
                    "edf_state_path": str(state_root / "edf" / split / f"{safe_id}.json"),
                    "state_key": state_key,
                }
                tasks.append(task)
                semantic_rows.append(
                    {
                        "recording_id": raw_recording_id,
                        "patient_id": patient_id,
                        "filename": filename,
                        "source_path": str(source_path),
                        "split": split,
                        "label": label,
                    }
                )
            except Exception as exc:
                failure_id = safe_recording_id(raw_recording_id or f"metadata_row_{row_number}")
                split_text = str(row.get("split", "unknown")).strip().lower()
                try:
                    split, source_split = normalise_split(split_text)
                except ValueError:
                    split, source_split = "unknown", split_text or "unknown"
                failure = {
                    **base,
                    "source_edf_path": "",
                    "split": split,
                    "source_split": source_split,
                    "label": "",
                    "label_name": "",
                    "status": "Metadata/EDF resolution error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pipeline_fingerprint": "",
                    "completed_at_utc": utc_now(),
                    "edf_state_path": str(state_root / "edf" / split / f"{failure_id}.json"),
                }
                failures.append(failure)

    semantic_rows.sort(key=lambda item: (item["recording_id"], item["filename"], item["split"]))
    metadata_fingerprint = semantic_json_hash({"records": semantic_rows})
    return tasks, failures, metadata_fingerprint


def make_pipeline_config(args: argparse.Namespace, metadata_fingerprint: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "metadata_fingerprint": metadata_fingerprint,
        "target_channels": list(TARGET_CHANNELS),
        "target_sampling_rate_hz": float(args.target_sfreq),
        "window_seconds": float(args.window_seconds),
        "low_hz": float(args.low_hz),
        "high_hz": float(args.high_hz),
        "notch_hz": float(args.notch_hz),
        "notch_q": float(args.notch_q),
        "filter_order": int(args.filter_order),
        "dtype": str(args.dtype),
        "pickle_protocol": int(pickle.HIGHEST_PROTOCOL),
        "units": "microvolts",
        "window_overlap_seconds": 0.0,
    }
    config["pipeline_fingerprint"] = semantic_json_hash(config)
    return config


def output_has_previous_state(output_root: Path, state_root: Path) -> bool:
    for split in ("train", "eval"):
        folder = output_root / split
        if folder.exists() and next(folder.glob("*.pkl"), None) is not None:
            return True
    if state_root.exists() and next(state_root.rglob("*.json*"), None) is not None:
        return True
    return False


def initialise_run_state(
    output_root: Path,
    state_root: Path,
    config: Mapping[str, Any],
    resume: bool,
) -> None:
    config_path = state_root / "pipeline_config.json"
    has_state = output_has_previous_state(output_root, state_root)

    if not resume and has_state:
        raise RuntimeError(
            f"Output directory already contains NMT windows/state: {output_root}\n"
            "For a clean restart, choose a new --output-root. To continue this exact run, add --resume."
        )

    if resume and has_state and not config_path.is_file():
        raise RuntimeError(
            f"{output_root} contains legacy/untracked files but no pipeline_config.json. "
            "Use a new empty --output-root so old and new windows cannot be mixed."
        )

    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("pipeline_fingerprint") != config.get("pipeline_fingerprint"):
            raise RuntimeError(
                "The preprocessing settings or metadata changed since this run began. "
                "Use a new --output-root rather than mixing incompatible windows.\n"
                f"Saved fingerprint:   {saved.get('pipeline_fingerprint')}\n"
                f"Current fingerprint: {config.get('pipeline_fingerprint')}"
            )
    else:
        state_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, config)

    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "eval").mkdir(parents=True, exist_ok=True)


def source_file_signature(path: Path) -> Tuple[str, int, int]:
    stat = path.stat()
    signature = f"{stat.st_size}:{stat.st_mtime_ns}"
    return signature, int(stat.st_size), int(stat.st_mtime_ns)


def pick_strict_target_channels(raw: mne.io.BaseRaw) -> Tuple[int, List[str]]:
    original_channel_count = len(raw.ch_names)
    matches: Dict[str, List[str]] = {channel: [] for channel in TARGET_CHANNELS}
    for original in raw.ch_names:
        canonical = clean_channel_name(original)
        if canonical in matches:
            matches[canonical].append(original)

    missing = [channel for channel, originals in matches.items() if not originals]
    duplicates = {channel: originals for channel, originals in matches.items() if len(originals) > 1}
    if missing:
        raise ValueError(f"Missing required EEG channels: {', '.join(missing)}")
    if duplicates:
        details = "; ".join(f"{channel}={names}" for channel, names in duplicates.items())
        raise ValueError(f"Ambiguous duplicate EEG channels after alias mapping: {details}")

    selected_originals = [matches[channel][0] for channel in TARGET_CHANNELS]
    rename_map = {original: canonical for original, canonical in zip(selected_originals, TARGET_CHANNELS)}
    raw.pick_channels(selected_originals, ordered=False)
    raw.rename_channels(rename_map)
    raw.reorder_channels(list(TARGET_CHANNELS))
    if tuple(raw.ch_names) != TARGET_CHANNELS:
        raise ValueError(f"Channel order mismatch after selection: {raw.ch_names}")
    return original_channel_count, selected_originals


def get_microvolt_data(raw: mne.io.BaseRaw) -> np.ndarray:
    try:
        data = raw.get_data(units="uV")
    except TypeError:  # Compatibility fallback for older MNE releases.
        data = raw.get_data() * 1_000_000.0
    return np.asarray(data, dtype=np.float64, order="C")


def preprocess_signal(
    data_uv: np.ndarray,
    source_sfreq: float,
    config: Mapping[str, Any],
) -> Tuple[np.ndarray, float, Optional[float]]:
    low_hz = float(config["low_hz"])
    requested_high = float(config["high_hz"])
    notch_hz = float(config["notch_hz"])
    nyquist = source_sfreq / 2.0
    safety_margin = max(0.01, nyquist * 0.001)
    effective_high = min(requested_high, nyquist - safety_margin)
    if not 0.0 < low_hz < effective_high:
        raise ValueError(
            f"Invalid band-pass for source sampling rate {source_sfreq:g} Hz: "
            f"low={low_hz:g}, effective high={effective_high:g}"
        )

    # SciPy SOS filtering is used directly here.  This avoids the old MNE 1.4
    # FIR wrapper depending on APIs removed from newer SciPy releases.
    sos = butter(
        int(config["filter_order"]),
        [low_hz, effective_high],
        btype="bandpass",
        fs=source_sfreq,
        output="sos",
    )
    filtered = sosfiltfilt(sos, data_uv, axis=-1)

    notch_applied: Optional[float] = None
    if notch_hz > 0.0 and notch_hz < nyquist - safety_margin:
        b_notch, a_notch = iirnotch(notch_hz, float(config["notch_q"]), fs=source_sfreq)
        filtered = filtfilt(b_notch, a_notch, filtered, axis=-1)
        notch_applied = notch_hz

    target_sfreq = float(config["target_sampling_rate_hz"])
    if not math.isclose(source_sfreq, target_sfreq, rel_tol=0.0, abs_tol=1e-9):
        ratio = Fraction(str(target_sfreq / source_sfreq)).limit_denominator(10000)
        filtered = resample_poly(filtered, ratio.numerator, ratio.denominator, axis=-1)

    dtype = np.float32 if config["dtype"] == "float32" else np.float64
    result = np.asarray(filtered, dtype=dtype, order="C")
    if result.ndim != 2 or result.shape[0] != len(TARGET_CHANNELS):
        raise ValueError(f"Unexpected preprocessed signal shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Preprocessed signal contains NaN or infinity")
    return result, effective_high, notch_applied


def validate_loaded_window(
    value: Any,
    expected_shape: Tuple[int, int],
    expected_label: int,
    expected_dtype: np.dtype,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("Pickle payload is not a dictionary")
    if set(value) != {"X", "y"}:
        raise ValueError(f"Pickle keys must be exactly X and y; got {sorted(value)}")
    x = value["X"]
    if not isinstance(x, np.ndarray):
        raise ValueError(f"X is not a numpy.ndarray: {type(x).__name__}")
    if x.shape != expected_shape:
        raise ValueError(f"X shape is {x.shape}, expected {expected_shape}")
    if x.dtype != expected_dtype:
        raise ValueError(f"X dtype is {x.dtype}, expected {expected_dtype}")
    if not x.flags.c_contiguous:
        raise ValueError("X is not C-contiguous")
    if not np.isfinite(x).all():
        raise ValueError("X contains NaN or infinity")
    if isinstance(value["y"], np.generic):
        actual_label = int(value["y"].item())
    else:
        actual_label = int(value["y"])
    if actual_label != expected_label:
        raise ValueError(f"Label is {actual_label}, expected {expected_label}")


def write_and_verify_window(
    path: Path,
    x: np.ndarray,
    label: int,
) -> Tuple[int, str]:
    payload = pickle.dumps({"X": x, "y": int(label)}, protocol=pickle.HIGHEST_PROTOCOL)
    expected_sha = hashlib.sha256(payload).hexdigest()
    atomic_write_bytes(path, payload)

    try:
        disk_payload = path.read_bytes()
        actual_sha = hashlib.sha256(disk_payload).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch after write: {actual_sha} != {expected_sha}")
        loaded = pickle.loads(disk_payload)
        validate_loaded_window(loaded, x.shape, int(label), x.dtype)
        return len(disk_payload), actual_sha
    except Exception:
        # This file was created by the current call and failed its mandatory
        # validation, so it must not remain available to the training loader.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(os.path.abspath(str(right)))


def fully_validate_existing_window(
    path: Path,
    row: Mapping[str, Any],
    expected_shape: Tuple[int, int],
    expected_label: int,
    expected_dtype: np.dtype,
) -> bool:
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != str(row.get("sha256", "")):
            return False
        value = pickle.loads(payload)
        validate_loaded_window(value, expected_shape, expected_label, expected_dtype)
        return True
    except Exception:
        return False


def reusable_window_row(
    row: Optional[Mapping[str, Any]],
    final_path: Path,
    index: int,
    source_signature: str,
    config: Mapping[str, Any],
    label: int,
    expected_shape: Tuple[int, int],
    resume_check: str,
) -> bool:
    if not row:
        return False
    try:
        if row.get("status") != "Valid":
            return False
        if int(row.get("window_index", -1)) != index:
            return False
        if int(row.get("label", -1)) != label:
            return False
        if str(row.get("source_signature", "")) != source_signature:
            return False
        if str(row.get("pipeline_fingerprint", "")) != config["pipeline_fingerprint"]:
            return False
        if not same_path(Path(str(row.get("file_path", ""))), final_path):
            return False
        if not final_path.is_file():
            return False
        if final_path.stat().st_size != int(row.get("file_size_bytes", -1)):
            return False
        if resume_check == "full":
            dtype = np.dtype(config["dtype"])
            return fully_validate_existing_window(final_path, row, expected_shape, label, dtype)
        return True
    except (OSError, TypeError, ValueError):
        return False


def base_edf_summary(task: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_row_number": task.get("source_row_number", ""),
        "source_edf_path": task.get("source_edf_path", ""),
        "source_edf_name": task.get("source_edf_name", ""),
        "metadata_file_path": task.get("metadata_file_path", ""),
        "recording_id": task.get("recording_id", ""),
        "patient_id": task.get("patient_id", ""),
        "split": task.get("split", ""),
        "source_split": task.get("source_split", ""),
        "label": task.get("label", ""),
        "label_name": task.get("label_name", ""),
        "pipeline_fingerprint": config.get("pipeline_fingerprint", ""),
    }


def try_fast_resume(
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    resume_check: str,
    source_signature: str,
    existing_rows: Mapping[int, Mapping[str, Any]],
    malformed_lines: int,
    started: float,
) -> Optional[Dict[str, Any]]:
    """Skip EDF loading/filtering when a prior complete record is still intact."""
    edf_state_path = Path(str(task["edf_state_path"]))
    if not edf_state_path.is_file():
        return None
    try:
        with edf_state_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("status") != "Complete":
            return None
        if saved.get("pipeline_fingerprint") != config["pipeline_fingerprint"]:
            return None
        if saved.get("source_signature") != source_signature:
            return None
        if int(saved.get("label", -1)) != int(task["label"]):
            return None
        if str(saved.get("split", "")) != str(task["split"]):
            return None

        target_sfreq = float(config["target_sampling_rate_hz"])
        window_samples_float = target_sfreq * float(config["window_seconds"])
        window_samples = int(round(window_samples_float))
        if not math.isclose(window_samples, window_samples_float, abs_tol=1e-9):
            return None
        expected_shape = (len(TARGET_CHANNELS), window_samples)
        number_of_windows = int(saved.get("windows_expected", -1))
        if number_of_windows < 1:
            return None

        output_dir = Path(str(task["output_dir"]))
        for index in range(number_of_windows):
            final_path = output_dir / f"{task['safe_recording_id']}_{index}.pkl"
            if not reusable_window_row(
                existing_rows.get(index),
                final_path,
                index,
                source_signature,
                config,
                int(task["label"]),
                expected_shape,
                resume_check,
            ):
                return None

        resumed = dict(saved)
        resumed.update(base_edf_summary(task, config))
        resumed.update(
            {
                "status": "Complete",
                "error": "" if malformed_lines == 0 else f"Ignored {malformed_lines} torn state line(s)",
                "windows_written_this_run": 0,
                "windows_reused": number_of_windows,
                "processing_seconds": f"{time.perf_counter() - started:.3f}",
                "completed_at_utc": utc_now(),
            }
        )
        atomic_write_json(edf_state_path, resumed)
        return resumed
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def process_recording(payload: Tuple[Dict[str, Any], Dict[str, Any], str]) -> Dict[str, Any]:
    task, config, resume_check = payload
    started = time.perf_counter()
    summary = base_edf_summary(task, config)
    raw: Optional[mne.io.BaseRaw] = None
    try:
        mne.set_log_level("ERROR")
        source_path = Path(task["source_edf_path"])
        output_dir = Path(task["output_dir"])
        window_state_path = Path(task["window_state_path"])
        edf_state_path = Path(task["edf_state_path"])
        source_signature, _, _ = source_file_signature(source_path)

        existing_rows, malformed_lines = read_latest_jsonl_rows(window_state_path)
        resumed = try_fast_resume(
            task,
            config,
            resume_check,
            source_signature,
            existing_rows,
            malformed_lines,
            started,
        )
        if resumed is not None:
            return resumed

        raw = mne.io.read_raw_edf(str(source_path), preload=True, verbose="ERROR")
        source_sfreq = float(raw.info["sfreq"])
        original_channel_count, _ = pick_strict_target_channels(raw)
        data_uv = get_microvolt_data(raw)
        del raw
        raw = None

        processed, effective_high, notch_applied = preprocess_signal(data_uv, source_sfreq, config)
        del data_uv

        target_sfreq = float(config["target_sampling_rate_hz"])
        window_samples_float = target_sfreq * float(config["window_seconds"])
        window_samples = int(round(window_samples_float))
        if not math.isclose(window_samples, window_samples_float, abs_tol=1e-9):
            raise ValueError(
                f"target_sfreq * window_seconds must be an integer; got {window_samples_float}"
            )
        expected_shape = (len(TARGET_CHANNELS), window_samples)
        number_of_windows = processed.shape[1] // window_samples
        if number_of_windows < 1:
            raise ValueError(
                f"Recording is shorter than one {config['window_seconds']:g}-second window"
            )

        written = 0
        reused = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        dtype = np.dtype(config["dtype"])

        for index in range(number_of_windows):
            final_path = output_dir / f"{task['safe_recording_id']}_{index}.pkl"
            previous = existing_rows.get(index)
            if reusable_window_row(
                previous,
                final_path,
                index,
                source_signature,
                config,
                int(task["label"]),
                expected_shape,
                resume_check,
            ):
                reused += 1
                continue

            start_sample = index * window_samples
            end_sample = start_sample + window_samples
            x = np.ascontiguousarray(processed[:, start_sample:end_sample], dtype=dtype)
            if x.shape != expected_shape:
                raise ValueError(f"Window {index} has shape {x.shape}; expected {expected_shape}")
            file_size, digest = write_and_verify_window(final_path, x, int(task["label"]))
            row = {
                "file_path": str(final_path.resolve()),
                "status": "Valid",
                "shape_x": str(expected_shape),
                "label": int(task["label"]),
                "error": "",
                "split": task["split"],
                "source_split": task["source_split"],
                "label_name": task["label_name"],
                "recording_id": task["recording_id"],
                "patient_id": task["patient_id"],
                "window_index": index,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_sec": f"{start_sample / target_sfreq:.6f}",
                "end_sec": f"{end_sample / target_sfreq:.6f}",
                "source_edf_path": str(source_path.resolve()),
                "source_edf_name": source_path.name,
                "sampling_rate_hz": target_sfreq,
                "original_sampling_rate_hz": source_sfreq,
                "channels": "|".join(TARGET_CHANNELS),
                "dtype": str(dtype),
                "file_size_bytes": file_size,
                "sha256": digest,
                "validation": "atomic write + SHA-256 + reload + schema/shape/label/finiteness",
                "pipeline_fingerprint": config["pipeline_fingerprint"],
                "source_signature": source_signature,
                "created_at_utc": utc_now(),
            }
            append_jsonl(window_state_path, row)
            existing_rows[index] = row
            written += 1

        summary.update(
            {
                "status": "Complete",
                "error": "" if malformed_lines == 0 else f"Ignored {malformed_lines} torn state line(s)",
                "original_sampling_rate_hz": source_sfreq,
                "target_sampling_rate_hz": target_sfreq,
                "original_channel_count": original_channel_count,
                "selected_channel_count": len(TARGET_CHANNELS),
                "selected_channels": "|".join(TARGET_CHANNELS),
                "duration_sec_after_resampling": f"{processed.shape[1] / target_sfreq:.6f}",
                "samples_after_resampling": int(processed.shape[1]),
                "windows_expected": number_of_windows,
                "windows_written_this_run": written,
                "windows_reused": reused,
                "discarded_tail_samples": int(processed.shape[1] % window_samples),
                "effective_high_hz": f"{effective_high:.6f}",
                "notch_applied_hz": "" if notch_applied is None else notch_applied,
                "source_signature": source_signature,
                "processing_seconds": f"{time.perf_counter() - started:.3f}",
                "completed_at_utc": utc_now(),
            }
        )
        atomic_write_json(edf_state_path, summary)
        return summary
    except Exception as exc:
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
        summary.update(
            {
                "status": "Error",
                "error": f"{type(exc).__name__}: {exc}",
                "processing_seconds": f"{time.perf_counter() - started:.3f}",
                "completed_at_utc": utc_now(),
            }
        )
        try:
            atomic_write_json(Path(task["edf_state_path"]), summary)
        except Exception as state_exc:
            summary["error"] += f" | Could not save EDF state: {state_exc}"
        summary["traceback"] = traceback.format_exc(limit=8)
        return summary


def read_edf_state_files(state_root: Path) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    edf_root = state_root / "edf"
    if not edf_root.exists():
        return summaries
    for path in sorted(edf_root.rglob("*.json"), key=lambda item: str(item).lower()):
        try:
            with path.open("r", encoding="utf-8") as handle:
                summaries.append(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            summaries.append(
                {
                    "status": "State read error",
                    "error": f"{path}: {type(exc).__name__}: {exc}",
                    "completed_at_utc": utc_now(),
                }
            )
    return summaries


def iter_consolidated_window_rows(
    summaries: Sequence[Mapping[str, Any]],
    state_root: Path,
    config: Mapping[str, Any],
    counters: Counter,
) -> Iterator[Dict[str, Any]]:
    complete = [row for row in summaries if row.get("status") == "Complete"]
    complete.sort(key=lambda row: (str(row.get("split", "")), str(row.get("recording_id", ""))))

    for summary in complete:
        split = str(summary["split"])
        safe_id = safe_recording_id(str(summary["recording_id"]))
        state_path = state_root / "windows" / split / f"{safe_id}.jsonl"
        latest, malformed = read_latest_jsonl_rows(state_path)
        counters["malformed_state_lines"] += malformed
        expected = int(summary.get("windows_expected", 0))
        source_signature = str(summary.get("source_signature", ""))

        for index in range(expected):
            row = latest.get(index)
            if (
                row is None
                or row.get("pipeline_fingerprint") != config["pipeline_fingerprint"]
                or row.get("source_signature") != source_signature
            ):
                counters["invalid_windows"] += 1
                yield {
                    "file_path": "",
                    "status": "Missing metadata",
                    "shape_x": "",
                    "label": summary.get("label", ""),
                    "error": f"No valid state row for expected window {index}",
                    "split": split,
                    "source_split": summary.get("source_split", ""),
                    "label_name": summary.get("label_name", ""),
                    "recording_id": summary.get("recording_id", ""),
                    "patient_id": summary.get("patient_id", ""),
                    "window_index": index,
                    "source_edf_path": summary.get("source_edf_path", ""),
                    "source_edf_name": summary.get("source_edf_name", ""),
                    "pipeline_fingerprint": config["pipeline_fingerprint"],
                    "source_signature": source_signature,
                }
                continue

            output = dict(row)
            try:
                path = Path(str(row["file_path"]))
                if not path.is_file():
                    raise FileNotFoundError("pickle is missing after preprocessing")
                actual_size = path.stat().st_size
                expected_size = int(row["file_size_bytes"])
                if actual_size != expected_size:
                    raise ValueError(f"file size {actual_size}, expected {expected_size}")
                output["status"] = "Valid"
                output["error"] = ""
                counters["valid_windows"] += 1
            except Exception as exc:
                output["status"] = "Corrupted / Error"
                output["error"] = f"{type(exc).__name__}: {exc}"
                counters["invalid_windows"] += 1
            yield output


def save_preflight_failures(failures: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    for failure in failures:
        row = dict(failure)
        row["pipeline_fingerprint"] = config["pipeline_fingerprint"]
        atomic_write_json(Path(str(failure["edf_state_path"])), row)


def print_versions() -> None:
    print(f"Python: {platform.python_version()} ({platform.platform()})")
    print(f"MNE: {mne.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess NMT EDFs into verified LaBraM windows and provenance metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata-csv", type=Path, default=PROJECT_ROOT / "data" / "metadata.csv")
    parser.add_argument("--raw-root", type=Path, default=Path(r"F:\Dataset\NMT-4K-EEG"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"F:\NMT_processed_clean"),
        help="Use a new empty directory for a clean restart",
    )
    parser.add_argument(
        "--window-report",
        type=Path,
        default=PROJECT_ROOT / "data" / "windows_inspection_report.csv",
    )
    parser.add_argument(
        "--edf-report",
        type=Path,
        default=PROJECT_ROOT / "data" / "edf_processing_report.csv",
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 3))
    parser.add_argument("--target-sfreq", type=float, default=200.0)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--low-hz", type=float, default=0.1)
    parser.add_argument("--high-hz", type=float, default=75.0)
    parser.add_argument("--notch-hz", type=float, default=50.0, help="Set to 0 to disable")
    parser.add_argument("--notch-q", type=float, default=30.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a run created by this script; settings and metadata must match",
    )
    parser.add_argument(
        "--resume-check",
        choices=("size", "full"),
        default="size",
        help="How to check already verified windows while resuming; full rereads each one",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N resolved EDFs (0 means all; useful for a trial run)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.target_sfreq <= 0 or args.window_seconds <= 0:
        parser.error("--target-sfreq and --window-seconds must be positive")
    if args.low_hz <= 0 or args.high_hz <= args.low_hz:
        parser.error("require 0 < --low-hz < --high-hz")
    if args.filter_order < 1:
        parser.error("--filter-order must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    print_versions()
    print("\nIndexing raw EDF files...")
    edf_index = build_edf_index(args.raw_root)
    edf_count = sum(len(paths) for paths in edf_index.values())
    print(f"Found {edf_count:,} EDF file(s) under {args.raw_root}")

    state_root = args.output_root / ".nmt_preprocess_state"
    tasks, preflight_failures, metadata_fingerprint = load_metadata_tasks(
        args.metadata_csv,
        args.raw_root,
        args.output_root,
        state_root,
        edf_index,
    )
    config = make_pipeline_config(args, metadata_fingerprint)
    initialise_run_state(args.output_root, state_root, config, args.resume)
    removed_temps = clean_stale_temporary_files(args.output_root)
    save_preflight_failures(preflight_failures, config)

    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Resolved EDF records: {len(tasks):,}")
    print(f"Metadata/path errors: {len(preflight_failures):,}")
    print(f"Output root: {args.output_root}")
    print(f"Pipeline fingerprint: {config['pipeline_fingerprint']}")
    if removed_temps:
        print(f"Removed {removed_temps:,} stale temporary file(s) from an interrupted write.")

    results: List[Dict[str, Any]] = []
    if tasks:
        work = [(task, config, args.resume_check) for task in tasks]
        workers = min(args.workers, len(work))
        print(f"\nProcessing with {workers} worker process(es)...")
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(process_recording, work, chunksize=1)
            for result in tqdm(iterator, total=len(work), desc="Processing EDFs", unit="edf", ncols=100):
                results.append(result)
                if result.get("status") == "Error":
                    print(
                        f"\nERROR {result.get('source_edf_name')}: {result.get('error')}",
                        file=sys.stderr,
                    )

    summaries = read_edf_state_files(state_root)
    edf_rows = sorted(
        summaries,
        key=lambda row: (str(row.get("split", "")), str(row.get("recording_id", ""))),
    )
    edf_row_count = atomic_write_csv(args.edf_report, EDF_FIELDS, edf_rows)

    counters: Counter = Counter()
    window_rows = iter_consolidated_window_rows(summaries, state_root, config, counters)
    window_row_count = atomic_write_csv(args.window_report, WINDOW_FIELDS, window_rows)

    status_counts = Counter(str(row.get("status", "Unknown")) for row in summaries)
    print("\nPreprocessing summary")
    print(f"  EDF report rows:       {edf_row_count:,}")
    print(f"  Complete EDFs:         {status_counts.get('Complete', 0):,}")
    print(f"  EDF errors:            {sum(v for k, v in status_counts.items() if k != 'Complete'):,}")
    print(f"  Window report rows:    {window_row_count:,}")
    print(f"  Valid windows:         {counters['valid_windows']:,}")
    print(f"  Invalid/missing:       {counters['invalid_windows']:,}")
    print(f"  Window metadata:       {args.window_report}")
    print(f"  EDF/error report:      {args.edf_report}")

    problems = sum(v for k, v in status_counts.items() if k != "Complete") + counters["invalid_windows"]
    if problems:
        print(
            "\nCompleted with recorded errors. Review edf_processing_report.csv; only rows "
            "marked Valid in windows_inspection_report.csv should be used for training.",
            file=sys.stderr,
        )
        return 2

    print("\nAll published windows passed integrated write-time validation.")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
