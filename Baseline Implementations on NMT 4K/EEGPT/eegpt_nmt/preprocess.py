"""Convert NMT EDF recordings to native four-second EEGPT windows.

The output index has one row per recording. Each tensor file contains every
valid window for that recording, which permits recording-balanced sampling and
recording-level evaluation later.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import load_config, require_sections, resolve_path
from .constants import (
    CHANNEL_ALIASES,
    EEGPT_SCALP_CHANNELS,
    EEGPT_SFREQ,
    EEGPT_WINDOW_SAMPLES,
    NMT_INPUT_CHANNELS,
)


mne.set_log_level("ERROR")


def clean_channel_name(name: str) -> str:
    """Normalize an EDF label such as ``EEG T3-REF`` to ``T7``."""
    cleaned = re.sub(r"(?i)^eeg\s*", "", str(name).strip())
    cleaned = re.sub(r"(?i)([-_\s](REF|AVG|LE))$", "", cleaned).strip().upper()
    return CHANNEL_ALIASES.get(cleaned, cleaned)


def resolve_edf_path(row: pd.Series, dataset_root: Path | None) -> Path:
    """Resolve a recording without trusting machine-specific paths in the CSV."""
    recorded_path = Path(str(row.get("source_file_path", "")))
    if recorded_path.exists():
        return recorded_path.resolve()
    if dataset_root is None:
        raise FileNotFoundError(
            f"EDF path is invalid for {row['recording_id']} and paths.dataset_root is not configured."
        )

    label_directory = "abnormal" if int(row["label"]) == 1 else "normal"
    filename = f"{row['recording_id']}.edf"
    candidates = [
        dataset_root / str(row["official_split"]) / label_directory / "edf" / filename,
        dataset_root / str(row["official_split"]) / label_directory / filename,
        dataset_root / label_directory / "edf" / filename,
        dataset_root / label_directory / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # A recursive fallback helps when the extracted dataset has an extra parent
    # directory. Ambiguous matches are rejected instead of silently guessed.
    matches = list(dataset_root.rglob(filename))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(f"Multiple EDF files match {filename}: {matches[:5]}")
    raise FileNotFoundError(f"Could not locate {filename} beneath {dataset_root}")


def _replace_nonfinite(raw: mne.io.BaseRaw) -> int:
    """Replace rare NaN/Inf samples with that channel's finite median."""
    data = raw.get_data()
    bad_mask = ~np.isfinite(data)
    bad_count = int(bad_mask.sum())
    if bad_count:
        for channel_index in range(data.shape[0]):
            channel_bad = bad_mask[channel_index]
            if not channel_bad.any():
                continue
            finite_values = data[channel_index, ~channel_bad]
            replacement = float(np.median(finite_values)) if finite_values.size else 0.0
            data[channel_index, channel_bad] = replacement
        # The Raw object was preloaded, so replacing its backing array is safe.
        raw._data[:] = data  # noqa: SLF001 - intentional MNE in-memory repair
    return bad_count


def _preprocessing_signature(options: dict[str, Any]) -> dict[str, Any]:
    """Store enough information to detect stale tensors from another setup."""
    return {
        "channels": NMT_INPUT_CHANNELS,
        "reference_channels": EEGPT_SCALP_CHANNELS,
        "target_sfreq": float(options.get("target_sfreq", EEGPT_SFREQ)),
        "window_seconds": float(options.get("window_seconds", 4.0)),
        "stride_seconds": float(options.get("stride_seconds", 4.0)),
        "bandpass_hz": [float(options.get("low_cut_hz", 0.5)), float(options.get("high_cut_hz", 40.0))],
        "crop_start_seconds": float(options.get("crop_start_seconds", 0.0)),
        "max_duration_seconds": options.get("max_duration_seconds"),
        "units": "microvolts",
    }


def preprocess_recording(
    edf_path: Path,
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read, filter, reference, resample, and slice one complete recording."""
    target_sfreq = float(options.get("target_sfreq", EEGPT_SFREQ))
    window_seconds = float(options.get("window_seconds", 4.0))
    stride_seconds = float(options.get("stride_seconds", window_seconds))
    window_samples = int(round(target_sfreq * window_seconds))
    stride_samples = int(round(target_sfreq * stride_seconds))
    if target_sfreq != EEGPT_SFREQ or window_samples != EEGPT_WINDOW_SAMPLES:
        raise ValueError(
            "This corrected baseline requires the native EEGPT input: 256 Hz and exactly 1,024 samples."
        )
    if stride_samples <= 0:
        raise ValueError("stride_seconds must be positive.")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    original_sfreq = float(raw.info["sfreq"])
    normalized_names = [clean_channel_name(name) for name in raw.ch_names]
    duplicates = sorted({name for name in normalized_names if normalized_names.count(name) > 1})
    required_duplicates = sorted(set(duplicates).intersection(NMT_INPUT_CHANNELS))
    if required_duplicates:
        raise ValueError(f"Duplicate required EEG channels after normalization: {required_duplicates}")
    # Rename only channels we plan to keep. Irrelevant auxiliary channels can
    # legitimately share generic labels, and renaming those could create an
    # unnecessary duplicate-name error in MNE.
    rename_map = {
        original: normalized
        for original, normalized in zip(raw.ch_names, normalized_names)
        if normalized in NMT_INPUT_CHANNELS and original != normalized
    }
    raw.rename_channels(rename_map)
    missing = sorted(set(NMT_INPUT_CHANNELS).difference(raw.ch_names))
    if missing:
        raise ValueError(f"Missing required channels: {missing}")
    # Raw.pick() does not accept ``ordered`` in several MNE releases. Pick and
    # reorder as two explicit operations for compatibility across versions.
    raw.pick(NMT_INPUT_CHANNELS)
    raw.reorder_channels(NMT_INPUT_CHANNELS)
    raw.set_channel_types({channel: "eeg" for channel in NMT_INPUT_CHANNELS}, verbose="ERROR")

    crop_start = float(options.get("crop_start_seconds", 0.0))
    max_duration = options.get("max_duration_seconds")
    if raw.n_times <= 0:
        raise ValueError(
            "EDF contains zero readable signal samples. The file is probably "
            "truncated, header-only, or has an invalid data offset; replace it "
            "with a verified copy rather than fabricating EEG samples."
        )
    recording_end = float(raw.times[-1])
    if recording_end - crop_start < window_seconds:
        raise ValueError(f"Recording is too short after crop: {recording_end - crop_start:.2f} seconds")
    crop_end = recording_end if max_duration in (None, "", 0, 0.0) else min(
        recording_end, crop_start + float(max_duration)
    )
    raw.crop(tmin=crop_start, tmax=crop_end, include_tmax=False)

    repaired_samples = _replace_nonfinite(raw)
    raw.filter(
        l_freq=float(options.get("low_cut_hz", 0.5)),
        h_freq=float(options.get("high_cut_hz", 40.0)),
        picks="eeg",
        method="fir",
        phase="zero-double",
        verbose="ERROR",
    )
    # Referencing to the 19 scalp electrodes prevents A1/A2 from changing the
    # common reference while retaining them as optional inputs to the adapter.
    raw.set_eeg_reference(ref_channels=EEGPT_SCALP_CHANNELS, projection=False, verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    try:
        data_uv = raw.get_data(units="uV").astype(np.float32, copy=False)
    except TypeError:
        # Older MNE versions do not expose the ``units`` keyword.
        data_uv = (raw.get_data() * 1e6).astype(np.float32, copy=False)

    starts = np.arange(0, data_uv.shape[1] - window_samples + 1, stride_samples, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("No full EEGPT windows remain after preprocessing.")
    windows = np.stack([data_uv[:, start : start + window_samples] for start in starts], axis=0)

    # Artifact rejection is deliberately optional. Aggressive automatic
    # removal can erase clinically meaningful high-amplitude abnormalities.
    keep = np.ones(len(windows), dtype=bool)
    max_abs_uv = options.get("reject_max_abs_uv")
    if max_abs_uv not in (None, "", 0, 0.0):
        keep &= np.max(np.abs(windows), axis=(1, 2)) <= float(max_abs_uv)
    flat_std_uv = options.get("reject_flat_std_uv")
    if flat_std_uv not in (None, "", 0, 0.0):
        keep &= np.mean(np.std(windows, axis=2), axis=1) >= float(flat_std_uv)
    rejected_windows = int((~keep).sum())
    windows = windows[keep]
    starts = starts[keep]
    if len(windows) == 0:
        raise ValueError("All windows were rejected by the configured artifact rules.")

    start_seconds = crop_start + starts.astype(np.float64) / target_sfreq
    quality = {
        "source_sfreq": original_sfreq,
        "nonfinite_samples_repaired": repaired_samples,
        "windows_before_rejection": int(len(keep)),
        "windows_rejected": rejected_windows,
    }
    return windows.astype(np.float16), start_seconds.astype(np.float32), quality


def _load_existing_count(tensor_path: Path, expected_signature: dict[str, Any]) -> int:
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "windows" not in payload:
        raise ValueError(f"Existing tensor is from the old format: {tensor_path}")
    if payload.get("preprocessing") != expected_signature:
        raise ValueError(
            f"Existing tensor used different preprocessing: {tensor_path}. Set preprocessing.overwrite=true."
        )
    return int(payload["windows"].shape[0])


def _portable_tensor_path(tensor_path: Path, project_root: Path) -> str:
    try:
        return tensor_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(tensor_path.resolve())


def run_preprocessing(config: dict[str, Any], retry_failures: bool = False) -> pd.DataFrame:
    """Preprocess every split row, write a full failure manifest, and verify counts."""
    paths = config["paths"]
    options = config["preprocessing"]
    project_root = Path(config["_project_root"])
    split_path = resolve_path(config, paths["recording_splits_csv"])
    output_directory = resolve_path(config, paths["processed_tensor_dir"])
    manifest_path = resolve_path(config, paths["processed_manifest_csv"])
    dataset_root = resolve_path(config, paths.get("dataset_root"))
    assert split_path is not None and output_directory is not None and manifest_path is not None
    if not split_path.exists():
        raise FileNotFoundError(f"Run prepare_splits first; missing: {split_path}")
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_splits = pd.read_csv(split_path)
    signature = _preprocessing_signature(options)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    overwrite = bool(options.get("overwrite", False))
    consecutive_failures = 0
    maximum_consecutive_failures = int(options.get("max_consecutive_failures", 10))
    failure_path = manifest_path.with_name("preprocessing_failures.csv")

    if retry_failures:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot retry failures because the successful-recording manifest is missing: {manifest_path}"
            )
        if not failure_path.exists():
            raise FileNotFoundError(
                f"Cannot retry failures because the failure manifest is missing: {failure_path}"
            )
        previous_failures = pd.read_csv(failure_path)
        if previous_failures.empty:
            print(f"No failed recording IDs remain in {failure_path}")
            return pd.read_csv(manifest_path)
        retry_ids = set(previous_failures["recording_id"].astype(str))
        splits = all_splits[all_splits["recording_id"].astype(str).isin(retry_ids)].copy()
        missing_retry_ids = sorted(retry_ids.difference(splits["recording_id"].astype(str)))
        if missing_retry_ids:
            raise ValueError(
                "Failure manifest contains recording IDs absent from recording_splits.csv: "
                f"{missing_retry_ids}"
            )
        existing_manifest = pd.read_csv(manifest_path)
        existing_manifest = existing_manifest[
            ~existing_manifest["recording_id"].astype(str).isin(retry_ids)
        ]
        records.extend(existing_manifest.to_dict(orient="records"))
        progress_description = "Retrying failed recordings"
    else:
        splits = all_splits
        progress_description = "Preprocessing recordings"

    for _, row in tqdm(splits.iterrows(), total=len(splits), desc=progress_description):
        recording_id = str(row["recording_id"])
        tensor_path = output_directory / f"{recording_id}.pt"
        edf_path: Path | None = None
        stage = "resolve_edf_path"
        try:
            edf_path = resolve_edf_path(row, dataset_root)
            if tensor_path.exists() and not overwrite:
                stage = "load_existing_tensor"
                number_of_windows = _load_existing_count(tensor_path, signature)
                quality = {"resumed_existing_tensor": True}
            else:
                stage = "read_and_preprocess_edf"
                windows, starts, quality = preprocess_recording(edf_path, options)
                payload = {
                    "windows": torch.from_numpy(windows),
                    "window_start_seconds": torch.from_numpy(starts),
                    "channels": NMT_INPUT_CHANNELS,
                    "label": int(row["label"]),
                    "recording_id": recording_id,
                    "preprocessing": signature,
                    "quality": quality,
                }
                stage = "save_tensor"
                temporary_path = tensor_path.with_suffix(".pt.tmp")
                torch.save(payload, temporary_path)
                os.replace(temporary_path, tensor_path)
                number_of_windows = int(len(windows))

            record = row.to_dict()
            record.update(
                {
                    "edf_path_resolved": str(edf_path),
                    "tensor_path": _portable_tensor_path(tensor_path, project_root),
                    "num_windows": number_of_windows,
                    "preprocessing_signature": json.dumps(signature, sort_keys=True),
                    "quality": json.dumps(quality, sort_keys=True),
                }
            )
            records.append(record)
            consecutive_failures = 0
        except Exception as error:  # Record all failures; strict mode raises after the audit files are saved.
            consecutive_failures += 1
            failures.append(
                {
                    "recording_id": recording_id,
                    "official_split": row.get("official_split"),
                    "experiment_split": row.get("experiment_split"),
                    "label": row.get("label"),
                    "stage": stage,
                    "edf_path_resolved": str(edf_path) if edf_path is not None else "",
                    "file_size_bytes": (
                        edf_path.stat().st_size
                        if edf_path is not None and edf_path.exists()
                        else ""
                    ),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
            tqdm.write(f"FAILED {recording_id}: {type(error).__name__}: {error}")
            if maximum_consecutive_failures > 0 and consecutive_failures >= maximum_consecutive_failures:
                pd.DataFrame(records).to_csv(manifest_path, index=False)
                pd.DataFrame(failures).to_csv(failure_path, index=False)
                raise RuntimeError(
                    f"Stopped after {consecutive_failures} consecutive preprocessing failures. "
                    f"Inspect {failure_path} before restarting."
                ) from error

    manifest = pd.DataFrame(records)
    if not manifest.empty:
        split_order = {
            recording_id: order
            for order, recording_id in enumerate(all_splits["recording_id"].astype(str))
        }
        manifest["_split_order"] = manifest["recording_id"].astype(str).map(split_order)
        manifest = manifest.sort_values("_split_order").drop(columns="_split_order").reset_index(drop=True)
    failure_columns = [
        "recording_id",
        "official_split",
        "experiment_split",
        "label",
        "stage",
        "edf_path_resolved",
        "file_size_bytes",
        "error_type",
        "error",
        "traceback",
    ]
    failure_frame = pd.DataFrame(failures, columns=failure_columns)
    manifest.to_csv(manifest_path, index=False)
    failure_frame.to_csv(failure_path, index=False)

    expected = config["split"]
    actual_train = int((manifest.get("official_split") == "train").sum()) if not manifest.empty else 0
    actual_eval = int((manifest.get("official_split") == "evaluation").sum()) if not manifest.empty else 0
    expected_train = int(expected.get("expected_train_recordings", actual_train))
    expected_eval = int(expected.get("expected_evaluation_recordings", actual_eval))
    cohort_ok = actual_train == expected_train and actual_eval == expected_eval and not failures

    print(f"\nProcessed recordings: {len(manifest)}")
    print(f"Failures: {len(failures)} (details: {failure_path})")
    print(f"Official train: {actual_train}/{expected_train}; evaluation: {actual_eval}/{expected_eval}")
    if bool(options.get("strict_complete_cohort", True)) and not cohort_ok:
        raise RuntimeError(
            "Preprocessing cohort is incomplete. Training is blocked until preprocessing_failures.csv is resolved."
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Process only IDs currently listed in preprocessing_failures.csv and merge successes into the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, ["paths", "split", "preprocessing"])
    run_preprocessing(config, retry_failures=args.retry_failures)


if __name__ == "__main__":
    main()
