from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from .channels import select_channel_indices
from .config import ensure_output_directories, load_config, resolved_config
from .cropping import (
    SHORT_FALLBACK_STRATEGY,
    STANDARD_CROP_STRATEGY,
    select_crop_bounds,
)
from .data import compute_window_starts, window_parameters
from .manifest import compare_expected_counts, count_records, discover_records
from .utils import atomic_write_json, print_heading, source_fingerprint, stable_hash

PIPELINE_VERSION = "nmt4k-multibknet-preprocess-v1"


def _preprocessing_signature(config: dict[str, Any]) -> str:
    signal_preprocessing = dict(config["preprocessing"])
    # This policy changes only how recordings that previously had no usable
    # window are handled. Excluding it keeps all standard v1 cache entries valid.
    signal_preprocessing.pop("short_recording_policy", None)
    relevant = {
        "pipeline_version": PIPELINE_VERSION,
        "channels": config["dataset"]["channels"],
        "preprocessing": signal_preprocessing,
    }
    return stable_hash(relevant)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.stem, suffix=".npy.tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _valid_cached(
    cache_path: Path,
    sidecar_path: Path,
    source: dict[str, int],
    signature: str,
    n_channels: int,
    short_recording_policy: str,
) -> dict[str, Any] | None:
    if not cache_path.exists() or not sidecar_path.exists():
        return None
    try:
        with sidecar_path.open("r", encoding="utf-8") as stream:
            sidecar = json.load(stream)
        if sidecar.get("source_fingerprint") != source:
            return None
        if sidecar.get("preprocessing_signature") != signature:
            return None
        array = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[0] != n_channels:
            return None
        if int(sidecar.get("n_samples", -1)) != int(array.shape[1]):
            return None
        if (
            sidecar.get("crop_strategy") == SHORT_FALLBACK_STRATEGY
            and short_recording_policy != "last_complete_window"
        ):
            return None
        return sidecar
    except Exception:  # noqa: BLE001 - any malformed cache must be rebuilt
        return None


def _process_recording(
    edf_path: Path,
    required_channels: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    preprocessing = config["preprocessing"]
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    try:
        indices, mapping = select_channel_indices(raw.ch_names, required_channels)
        original_sfreq = float(raw.info["sfreq"])
        raw_n_times = int(raw.n_times)
        start, stop, crop_strategy = select_crop_bounds(
            n_times=raw_n_times,
            sfreq=original_sfreq,
            crop_start_seconds=float(preprocessing["crop_start_seconds"]),
            max_duration_seconds=float(preprocessing["max_duration_seconds"]),
            window_seconds=float(preprocessing["window_seconds"]),
            short_recording_policy=str(
                preprocessing.get("short_recording_policy", "strict")
            ),
        )

        try:
            data = raw.get_data(picks=indices, start=start, stop=stop, units="uV")
        except TypeError:
            data = raw.get_data(picks=indices, start=start, stop=stop) * 1e6
    finally:
        raw.close()

    data = np.asarray(data, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("recording contains NaN or infinite signal values")

    clip = float(preprocessing["clip_microvolts"])
    np.clip(data, -clip, clip, out=data)
    if str(preprocessing["reference"]).lower() != "average":
        raise ValueError(
            "Only common-average reference is implemented for this paper protocol."
        )
    data -= data.mean(axis=0, keepdims=True)

    target_sfreq = float(preprocessing["target_sfreq"])
    if not np.isclose(original_sfreq, target_sfreq):
        info = mne.create_info(required_channels, sfreq=original_sfreq, ch_types="eeg")
        temporary_raw = mne.io.RawArray(data, info, verbose="ERROR")
        temporary_raw.resample(target_sfreq, npad="auto", verbose="ERROR")
        data = temporary_raw.get_data()
        temporary_raw.close()

    result = np.asarray(data, dtype=np.float32, order="C")
    if not np.isfinite(result).all():
        raise ValueError("preprocessed recording contains NaN or infinite values")
    metadata = {
        "original_sfreq": original_sfreq,
        "target_sfreq": target_sfreq,
        "raw_n_times": raw_n_times,
        "raw_duration_seconds": float(raw_n_times / original_sfreq),
        "raw_samples_used": int(stop - start),
        "crop_start_sample_used": int(start),
        "crop_start_seconds_used": float(start / original_sfreq),
        "crop_strategy": crop_strategy,
        "short_recording_fallback_used": crop_strategy == SHORT_FALLBACK_STRATEGY,
        "n_channels": int(result.shape[0]),
        "n_samples": int(result.shape[1]),
        "duration_seconds": float(result.shape[1] / target_sfreq),
        "channel_mapping": mapping,
    }
    return result, metadata


def preprocess_dataset(config_path: str | Path, overwrite: bool = False) -> int:
    config, paths = load_config(config_path)
    ensure_output_directories(paths)
    records = discover_records(paths)
    actual_counts = count_records(records)
    count_errors = compare_expected_counts(
        actual_counts, config["dataset"]["expected_counts"]
    )
    if count_errors:
        raise RuntimeError(
            "Dataset counts do not match the configured release:\n"
            + "\n".join(count_errors)
        )

    required_channels = list(config["dataset"]["channels"])
    short_recording_policy = str(
        config["preprocessing"].get("short_recording_policy", "strict")
    ).lower()
    signature = _preprocessing_signature(config)
    window_size, stride, include_final = window_parameters(config)
    mne.set_log_level("ERROR")

    print_heading("Resumable NMT-4K preprocessing")
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped = 0

    for row in tqdm(
        records.itertuples(index=False), total=len(records), unit="recording"
    ):
        edf_path = Path(row.edf_path)
        cache_path = (
            paths.cache_dir / row.split / row.label_name / f"{row.recording_id}.npy"
        )
        sidecar_path = cache_path.with_suffix(".json")
        source = source_fingerprint(edf_path)
        cached = (
            None
            if overwrite
            else _valid_cached(
                cache_path,
                sidecar_path,
                source,
                signature,
                len(required_channels),
                short_recording_policy,
            )
        )

        base = row._asdict()
        try:
            if cached is None:
                array, signal_metadata = _process_recording(
                    edf_path, required_channels, config
                )
                starts = compute_window_starts(
                    int(array.shape[1]), window_size, stride, include_final
                )
                if not starts:
                    raise ValueError("preprocessed signal contains no complete windows")
                _atomic_save_npy(cache_path, array)
                cached = {
                    **signal_metadata,
                    "n_windows": len(starts),
                    "cache_path": str(cache_path.resolve()),
                    "source_fingerprint": source,
                    "preprocessing_signature": signature,
                    "pipeline_version": PIPELINE_VERSION,
                }
                atomic_write_json(sidecar_path, cached)
                status = "processed"
            else:
                skipped += 1
                status = "cached"

            successes.append(
                {
                    **base,
                    "status": status,
                    "cache_path": str(cache_path.resolve()),
                    "n_channels": int(cached["n_channels"]),
                    "n_samples": int(cached["n_samples"]),
                    "n_windows": int(cached["n_windows"]),
                    "duration_seconds": float(cached["duration_seconds"]),
                    "raw_duration_seconds": float(
                        cached.get("raw_duration_seconds", float("nan"))
                    ),
                    "original_sfreq": float(cached["original_sfreq"]),
                    "target_sfreq": float(cached["target_sfreq"]),
                    "crop_strategy": str(
                        cached.get("crop_strategy", STANDARD_CROP_STRATEGY)
                    ),
                    "crop_start_seconds_used": float(
                        cached.get(
                            "crop_start_seconds_used",
                            config["preprocessing"]["crop_start_seconds"],
                        )
                    ),
                    "short_recording_fallback_used": bool(
                        cached.get("short_recording_fallback_used", False)
                    ),
                    "preprocessing_signature": signature,
                }
            )
        except Exception as exc:  # noqa: BLE001 - continue and write a file-level audit
            failures.append(
                {
                    **base,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    processed_manifest = pd.DataFrame(successes)
    failure_frame = pd.DataFrame(failures)
    manifest_path = paths.audit_dir / "processed_manifest.csv"
    failure_path = paths.audit_dir / "preprocessing_failures.csv"
    adaptation_path = paths.audit_dir / "short_recording_adaptations.csv"
    processed_manifest.to_csv(manifest_path, index=False)
    failure_frame.to_csv(failure_path, index=False)
    adaptation_columns = [
        "recording_id",
        "split",
        "label_name",
        "edf_path",
        "raw_duration_seconds",
        "duration_seconds",
        "crop_start_seconds_used",
        "crop_strategy",
    ]
    adaptations = processed_manifest.loc[
        processed_manifest.get(
            "short_recording_fallback_used",
            pd.Series(False, index=processed_manifest.index),
        ).astype(bool),
        [column for column in adaptation_columns if column in processed_manifest],
    ]
    adaptations.to_csv(adaptation_path, index=False)

    evaluation_failures = sum(row["split"] == "evaluation" for row in failures)
    training_failures = sum(row["split"] == "train" for row in failures)
    blocked = False
    if evaluation_failures and bool(
        config["preprocessing"]["fail_on_evaluation_error"]
    ):
        blocked = True
    if training_failures and not bool(
        config["preprocessing"]["allow_training_exclusions"]
    ):
        blocked = True

    summary = {
        "status": "blocking_errors" if blocked else "complete",
        "recordings_discovered": len(records),
        "recordings_successful": len(successes),
        "recordings_processed_this_run": int(len(successes) - skipped),
        "recordings_reused_from_cache": int(skipped),
        "training_failures": int(training_failures),
        "evaluation_failures": int(evaluation_failures),
        "short_recording_fallbacks": int(len(adaptations)),
        "total_windows": int(processed_manifest["n_windows"].sum())
        if len(processed_manifest)
        else 0,
        "channels": required_channels,
        "window_samples": window_size,
        "stride_samples": stride,
        "preprocessing_signature": signature,
        "manifest": str(manifest_path),
        "failures": str(failure_path),
        "short_recording_adaptations": str(adaptation_path),
        "resolved_config": resolved_config(config, paths),
    }
    atomic_write_json(paths.audit_dir / "preprocessing_summary.json", summary)

    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "resolved_config"},
            indent=2,
        )
    )
    return 1 if blocked else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess NMT-4K for Multi-BK-Net.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild valid cached recordings."
    )
    args = parser.parse_args()
    raise SystemExit(preprocess_dataset(args.config, args.overwrite))


if __name__ == "__main__":
    main()
