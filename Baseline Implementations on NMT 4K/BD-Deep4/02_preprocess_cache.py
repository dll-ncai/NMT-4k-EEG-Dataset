from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from nmt_deep4.common import ensure_dirs, get_paths, load_config
from nmt_deep4.data import SCALP_CHANNELS

mne.set_log_level("ERROR")

FLOAT16_MAX = float(np.finfo(np.float16).max)  # 65504.0
CACHE_CHECK_CHUNK_SAMPLES = 200_000
STATS_CHUNK_SAMPLES = 200_000


def normalize_channel_name(name: str) -> str:
    x = name.upper().strip()
    x = re.sub(r"^EEG\s*", "", x)
    x = x.replace(" ", "")
    for suffix in ("-REF", "-LE", "-AVG", "-A1", "-A2"):
        if x.endswith(suffix):
            x = x[: -len(suffix)]
    aliases = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    return aliases.get(x, x)


def find_actual_channels(raw) -> list[str]:
    norm_to_actual = {}
    for name in raw.ch_names:
        norm_to_actual.setdefault(normalize_channel_name(name), name)

    actual = []
    missing = []
    for canonical in SCALP_CHANNELS:
        key = canonical.upper()
        if key not in norm_to_actual:
            missing.append(canonical)
        else:
            actual.append(norm_to_actual[key])

    if missing:
        raise ValueError(f"Missing scalp channels {missing}; found {raw.ch_names}")
    return actual


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Write a .npy atomically so an interruption cannot leave a partial cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


def requested_cache_dtype(cfg: dict) -> np.dtype:
    name = str(cfg["preprocessing"].get("cache_dtype", "float16")).strip().lower()
    if name in {"float16", "fp16", "half"}:
        return np.dtype(np.float16)
    if name in {"float32", "fp32", "single"}:
        return np.dtype(np.float32)
    raise ValueError(
        f"Unsupported preprocessing.cache_dtype={name!r}. "
        "Use 'float16' or 'float32'."
    )


def validate_existing_cache(path: Path, cfg: dict) -> dict:
    """Validate an existing cache, including finiteness.

    This intentionally catches cache files produced by the old script where a
    float32 -> float16 cast could overflow to +/-inf. Valid float32 cache files
    are accepted when float16 is requested because the updated preprocessor can
    automatically promote only extreme recordings to float32.
    """
    arr = np.load(path, mmap_mode="r")

    if arr.ndim != 2:
        raise ValueError(f"expected 2D [channels,time], got shape={arr.shape}")
    if arr.shape[0] != len(SCALP_CHANNELS):
        raise ValueError(
            f"expected {len(SCALP_CHANNELS)} channels, got shape={arr.shape}"
        )
    if arr.shape[1] <= 0:
        raise ValueError("cache contains zero time samples")
    if arr.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError(f"unexpected cache dtype {arr.dtype}")

    requested = requested_cache_dtype(cfg)
    # If the user explicitly switches the config to float32, rebuild old
    # float16 caches so the requested precision is actually honored.
    if requested == np.dtype(np.float32) and arr.dtype != np.dtype(np.float32):
        raise ValueError(
            f"cache dtype is {arr.dtype}, but config now requests float32"
        )

    max_abs = 0.0
    for start in range(0, arr.shape[1], CACHE_CHECK_CHUNK_SAMPLES):
        block = np.asarray(
            arr[:, start : start + CACHE_CHECK_CHUNK_SAMPLES], dtype=np.float32
        )
        if not np.isfinite(block).all():
            raise ValueError("cache contains NaN or +/-Inf")
        if block.size:
            block_max = float(np.max(np.abs(block)))
            if block_max > max_abs:
                max_abs = block_max

    return {
        "n_samples": int(arr.shape[1]),
        "stored_dtype": str(arr.dtype),
        "max_abs_uv": max_abs,
    }


def preprocess_one(edf_path: Path, out_path: Path, cfg: dict) -> dict:
    pcfg = cfg["preprocessing"]
    target_sfreq = float(pcfg.get("sfreq", 100.0))
    l_freq = float(pcfg.get("l_freq", 0.5))
    h_freq = float(pcfg.get("h_freq", 40.0))
    clip_cfg = pcfg.get("clip_uv", None)
    clip_uv = None if clip_cfg is None else float(clip_cfg)
    max_minutes = pcfg.get("max_duration_minutes", None)

    # New option. No config change is required: "promote" is the safe default.
    # promote = preserve the extreme recording by saving that file as float32
    # clip    = saturate only at the float16 numerical limit (+/-65504 uV)
    # error   = stop and make the user inspect the recording
    overflow_mode = str(pcfg.get("float16_overflow", "promote")).strip().lower()
    if overflow_mode not in {"promote", "clip", "error"}:
        raise ValueError(
            "preprocessing.float16_overflow must be one of: "
            "'promote', 'clip', 'error'"
        )

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    actual = find_actual_channels(raw)
    raw.pick(actual)
    raw.reorder_channels(actual)

    if max_minutes is not None:
        max_sec = float(max_minutes) * 60.0
        if raw.times[-1] > max_sec:
            raw.crop(tmin=0.0, tmax=max_sec, include_tmax=False)

    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    # MNE returns EEG data in SI units (volts). Convert to microvolts.
    data_uv = raw.get_data().astype(np.float32, copy=False) * np.float32(1e6)

    # Count before sanitizing so unusual source data are visible in QC output.
    nonfinite_before = int(data_uv.size - np.count_nonzero(np.isfinite(data_uv)))
    if nonfinite_before:
        data_uv = np.nan_to_num(data_uv, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs_before_clip = (
        float(np.max(np.abs(data_uv))) if data_uv.size else 0.0
    )

    user_clipped_samples = 0
    if clip_uv is not None:
        if clip_uv <= 0:
            raise ValueError("preprocessing.clip_uv must be positive or null")
        user_clipped_samples = int(np.count_nonzero(np.abs(data_uv) > clip_uv))
        np.clip(data_uv, -clip_uv, clip_uv, out=data_uv)

    max_abs_after_user_clip = (
        float(np.max(np.abs(data_uv))) if data_uv.size else 0.0
    )

    requested = requested_cache_dtype(cfg)
    stored_dtype = requested
    auto_promoted = False
    float16_limit_clipped_samples = 0

    if requested == np.dtype(np.float16) and max_abs_after_user_clip > FLOAT16_MAX:
        if overflow_mode == "promote":
            # Preserve the signal values. Only this recording becomes float32;
            # the rest of the dataset can remain float16 to save disk space.
            stored_dtype = np.dtype(np.float32)
            auto_promoted = True
        elif overflow_mode == "clip":
            float16_limit_clipped_samples = int(
                np.count_nonzero(np.abs(data_uv) > FLOAT16_MAX)
            )
            np.clip(data_uv, -FLOAT16_MAX, FLOAT16_MAX, out=data_uv)
            stored_dtype = np.dtype(np.float16)
        else:  # error
            raise OverflowError(
                f"Filtered EEG reaches {max_abs_after_user_clip:.3f} uV, "
                f"which exceeds float16 range (+/-{FLOAT16_MAX:.0f} uV). "
                "Use cache_dtype: float32, set float16_overflow: promote, "
                "or inspect/clip the recording explicitly."
            )

    # The old script cast directly here, which could silently create +/-inf.
    # The logic above guarantees the cast is numerically representable.
    with np.errstate(over="raise", invalid="raise"):
        to_save = data_uv.astype(stored_dtype, copy=False)

    if not np.isfinite(to_save).all():
        raise FloatingPointError("Non-finite values remain immediately before cache save")

    atomic_save_npy(out_path, to_save)

    return {
        "n_samples": int(data_uv.shape[1]),
        "sfreq": float(raw.info["sfreq"]),
        "stored_dtype": str(stored_dtype),
        "max_abs_uv": max_abs_after_user_clip,
        "max_abs_uv_before_user_clip": max_abs_before_clip,
        "nonfinite_source_samples": nonfinite_before,
        "user_clipped_samples": user_clipped_samples,
        "float16_limit_clipped_samples": float16_limit_clipped_samples,
        "auto_promoted_float32": bool(auto_promoted),
    }


def compute_train_stats(df: pd.DataFrame, stats_path: Path) -> None:
    """Compute per-channel mean/std using model-fitting recordings only."""
    train = df[df["role"] == "train"].reset_index(drop=True)
    sums = np.zeros(len(SCALP_CHANNELS), dtype=np.float64)
    sumsq = np.zeros(len(SCALP_CHANNELS), dtype=np.float64)
    counts = np.zeros(len(SCALP_CHANNELS), dtype=np.int64)

    for row in tqdm(
        train.itertuples(index=False),
        total=len(train),
        desc="Computing train-only channel stats",
    ):
        x = np.load(row.cache_path, mmap_mode="r")
        if x.ndim != 2 or x.shape[0] != len(SCALP_CHANNELS) or x.shape[1] <= 0:
            raise ValueError(f"Invalid cache during stats: {row.cache_path}, shape={x.shape}")

        for start in range(0, x.shape[1], STATS_CHUNK_SAMPLES):
            block = np.asarray(
                x[:, start : start + STATS_CHUNK_SAMPLES], dtype=np.float32
            )
            if not np.isfinite(block).all():
                raise ValueError(
                    f"Non-finite values found in cache during stats: {row.cache_path}. "
                    "Rerun preprocessing with the updated script."
                )

            b64 = block.astype(np.float64, copy=False)
            sums += b64.sum(axis=1, dtype=np.float64)
            sumsq += np.square(b64).sum(axis=1, dtype=np.float64)
            counts += block.shape[1]

    if np.any(counts == 0):
        raise RuntimeError("At least one EEG channel has zero samples while computing stats")

    mean = sums / counts
    var = np.maximum(sumsq / counts - np.square(mean), 1e-8)
    std = np.sqrt(var)

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError(
            "Non-finite normalization statistics were produced. "
            "Inspect preprocess_qc.csv for extreme-amplitude recordings."
        )

    np.savez(
        stats_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        channels=np.array(SCALP_CHANNELS),
    )
    print("Saved train-only normalization stats:", stats_path)
    for ch, m, s in zip(SCALP_CHANNELS, mean, std):
        print(f"  {ch:>3s}: mean={m:10.3f} uV, std={s:10.3f} uV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild every cache file, including valid existing ones.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ensure_dirs(paths)

    if not paths["manifest"].exists():
        raise FileNotFoundError("Run 01_prepare_manifest.py first.")

    df = pd.read_csv(paths["manifest"])
    failures: list[tuple[str, str, str]] = []
    qc_rows: list[dict] = []
    n_samples: list[float] = []
    sfreqs: list[float] = []

    repaired_bad_cache = 0
    promoted_float32 = 0
    skipped_valid = 0

    pbar = tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Preprocessing EDF -> NPY",
    )

    for row in pbar:
        out = Path(row.cache_path)
        must_rebuild = bool(args.overwrite)
        rebuild_reason = "--overwrite" if args.overwrite else ""

        if out.exists() and not must_rebuild:
            try:
                info = validate_existing_cache(out, cfg)
                n_samples.append(info["n_samples"])
                sfreqs.append(float(cfg["preprocessing"].get("sfreq", 100.0)))
                skipped_valid += 1

                qc_rows.append(
                    {
                        "file_name": row.file_name,
                        "edf_path": row.edf_path,
                        "cache_path": str(out),
                        "status": "existing_valid",
                        "stored_dtype": info["stored_dtype"],
                        "max_abs_uv": info["max_abs_uv"],
                        "max_abs_uv_before_user_clip": np.nan,
                        "nonfinite_source_samples": np.nan,
                        "user_clipped_samples": np.nan,
                        "float16_limit_clipped_samples": np.nan,
                        "auto_promoted_float32": (
                            info["stored_dtype"] == "float32"
                            and requested_cache_dtype(cfg) == np.dtype(np.float16)
                        ),
                        "rebuild_reason": "",
                    }
                )
                pbar.set_postfix(
                    skipped=skipped_valid,
                    repaired=repaired_bad_cache,
                    promoted=promoted_float32,
                )
                continue
            except Exception as e:
                must_rebuild = True
                repaired_bad_cache += 1
                rebuild_reason = f"invalid existing cache: {e}"
                tqdm.write(
                    f"Repairing cache for {row.file_name}: {e}"
                )

        try:
            info = preprocess_one(Path(row.edf_path), out, cfg)
            n_samples.append(info["n_samples"])
            sfreqs.append(info["sfreq"])

            if info["auto_promoted_float32"]:
                promoted_float32 += 1
                tqdm.write(
                    f"Float16 overflow avoided for {row.file_name}: "
                    f"max |EEG|={info['max_abs_uv']:.3f} uV > {FLOAT16_MAX:.0f} uV; "
                    "stored this recording as float32 instead."
                )

            if info["nonfinite_source_samples"]:
                tqdm.write(
                    f"Warning: {row.file_name} had "
                    f"{info['nonfinite_source_samples']} non-finite source samples; "
                    "they were replaced with 0."
                )

            qc_rows.append(
                {
                    "file_name": row.file_name,
                    "edf_path": row.edf_path,
                    "cache_path": str(out),
                    "status": "rebuilt" if must_rebuild else "processed",
                    "stored_dtype": info["stored_dtype"],
                    "max_abs_uv": info["max_abs_uv"],
                    "max_abs_uv_before_user_clip": info[
                        "max_abs_uv_before_user_clip"
                    ],
                    "nonfinite_source_samples": info[
                        "nonfinite_source_samples"
                    ],
                    "user_clipped_samples": info["user_clipped_samples"],
                    "float16_limit_clipped_samples": info[
                        "float16_limit_clipped_samples"
                    ],
                    "auto_promoted_float32": info["auto_promoted_float32"],
                    "rebuild_reason": rebuild_reason,
                }
            )

        except Exception as e:
            failures.append((row.file_name, row.edf_path, repr(e)))
            n_samples.append(np.nan)
            sfreqs.append(np.nan)
            qc_rows.append(
                {
                    "file_name": row.file_name,
                    "edf_path": row.edf_path,
                    "cache_path": str(out),
                    "status": "failed",
                    "stored_dtype": "",
                    "max_abs_uv": np.nan,
                    "max_abs_uv_before_user_clip": np.nan,
                    "nonfinite_source_samples": np.nan,
                    "user_clipped_samples": np.nan,
                    "float16_limit_clipped_samples": np.nan,
                    "auto_promoted_float32": False,
                    "rebuild_reason": rebuild_reason,
                }
            )
            tqdm.write(f"FAILED {row.file_name}: {e!r}")

        pbar.set_postfix(
            skipped=skipped_valid,
            repaired=repaired_bad_cache,
            promoted=promoted_float32,
        )

    pbar.close()

    df["cache_n_samples"] = n_samples
    df["cache_sfreq"] = sfreqs
    df.to_csv(paths["manifest"], index=False)

    qc_path = paths["manifest_dir"] / "preprocess_qc.csv"
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(qc_path, index=False)
    print("Saved preprocessing QC report:", qc_path)

    if not qc_df.empty and "max_abs_uv" in qc_df:
        finite_amp = pd.to_numeric(qc_df["max_abs_uv"], errors="coerce")
        extreme = qc_df[finite_amp > 5000.0]
        if len(extreme):
            extreme_path = paths["manifest_dir"] / "extreme_amplitude_recordings.csv"
            extreme.to_csv(extreme_path, index=False)
            print(
                f"QC note: {len(extreme)} recording(s) exceeded 5000 uV after filtering. "
                f"Review: {extreme_path}"
            )

    if failures:
        report = paths["manifest_dir"] / "preprocess_failures.csv"
        pd.DataFrame(
            failures, columns=["file_name", "edf_path", "error"]
        ).to_csv(report, index=False)
        raise RuntimeError(
            f"Preprocessing failed for {len(failures)} recordings. See {report}"
        )

    print(
        "\nCache validation/repair summary:\n"
        f"  Valid caches skipped : {skipped_valid}\n"
        f"  Bad caches repaired  : {repaired_bad_cache}\n"
        f"  Auto-promoted to fp32: {promoted_float32}"
    )

    compute_train_stats(df, paths["stats"])

    print(
        "\nPreprocessing complete. The updated script checks existing cache files "
        "for NaN/Inf, so rerunning remains safe and resumable."
    )


if __name__ == "__main__":
    main()
