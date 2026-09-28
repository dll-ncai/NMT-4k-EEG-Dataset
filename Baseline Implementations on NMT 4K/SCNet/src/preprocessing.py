from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly


SCALP_19 = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
    "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2",
]
PAPER_21 = SCALP_19 + ["A1", "A2"]
BIPOLAR_22 = [
    ("FP1", "F7"), ("FP1", "F3"), ("FP2", "F4"), ("FP2", "F8"),
    ("F7", "T3"), ("F3", "C3"), ("F4", "C4"), ("F8", "T4"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"),
    ("C4", "T4"), ("T4", "A2"), ("T3", "T5"), ("C3", "P3"),
    ("C4", "P4"), ("T4", "T6"), ("T5", "O1"), ("P3", "O1"),
    ("P4", "O2"), ("T6", "O2"),
]


def _channel_key(name: str) -> str:
    """Normalize common EDF EEG channel naming variants."""
    s = name.strip().upper()
    s = s.replace("EEG", "").replace(" ", "").replace("_", "-")
    s = s.strip("-")
    # Common aliases from modern nomenclature to legacy 10-20 names used by SCNet.
    alias = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    if s in alias:
        return alias[s]
    # Remove typical reference suffixes while preserving standalone A1/A2.
    for suffix in ("-REF", "-LE", "-AVG", "-AR"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Some clinical exports use labels such as FP1-A1 / FP2-A2. For the NMT-4K
    # release the scalp channels are already linked-ear referenced, so the base
    # electrode name is what we need here.
    m = re.fullmatch(r"(FP1|FP2|F7|F3|FZ|F4|F8|T3|C3|CZ|C4|T4|T5|P3|PZ|P4|T6|O1|O2)-(A1|A2)", s)
    if m:
        s = m.group(1)
    return alias.get(s, s)


def resolve_channels(raw: mne.io.BaseRaw, required: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for actual in raw.ch_names:
        key = _channel_key(actual)
        if key not in mapping:
            mapping[key] = actual
    missing = [c for c in required if c not in mapping]
    if missing:
        raise ValueError(
            f"Missing required channels {missing}. Available EDF channels: {raw.ch_names}"
        )
    return {c: mapping[c] for c in required}


def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    nyq = sfreq / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid bandpass [{low}, {high}] for sfreq={sfreq} Hz")
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    # sosfiltfilt works channel-wise along the final axis.
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32, copy=False)


def _resample(data: np.ndarray, orig_sfreq: float, target_sfreq: float) -> np.ndarray:
    if np.isclose(orig_sfreq, target_sfreq):
        return data.astype(np.float32, copy=False)
    ratio = Fraction(target_sfreq / orig_sfreq).limit_denominator(1000)
    return resample_poly(data, up=ratio.numerator, down=ratio.denominator, axis=-1).astype(np.float32, copy=False)


def choose_window(duration_sec: float, discard_sec: float, input_sec: float) -> float:
    """Choose one deterministic fixed-length recording window.

    If enough data are available, discard the first minute exactly as SCNet did.
    If the recording is >= input length but < discard+input, use the latest full
    input window to avoid padding. Shorter recordings use data after the discard
    point when possible and are zero-padded later.
    """
    if duration_sec >= discard_sec + input_sec:
        return discard_sec
    if duration_sec >= input_sec:
        return max(0.0, duration_sec - input_sec)
    if duration_sec > discard_sec:
        return discard_sec
    return 0.0


def preprocess_edf(
    edf_path: str | Path,
    mode: str,
    target_sfreq: float,
    input_seconds: float,
    discard_seconds: float,
    bandpass: list[float] | None,
    clip_uv: float | None,
    cache_dtype: str = "float16",
) -> tuple[np.ndarray, dict[str, Any]]:
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    orig_sfreq = float(raw.info["sfreq"])
    duration = float(raw.n_times / orig_sfreq)

    if mode == "nmt4k19":
        required = SCALP_19
    elif mode == "paper22":
        required = PAPER_21
    else:
        raise ValueError("preprocess.mode must be 'nmt4k19' or 'paper22'")

    ch_map = resolve_channels(raw, required)
    picks = [ch_map[c] for c in required]

    start_sec = choose_window(duration, discard_seconds, input_seconds)
    # Add a short margin for stable filtering, then trim it away after filtering.
    margin = 5.0 if bandpass else 0.0
    read_start_sec = max(0.0, start_sec - margin)
    read_stop_sec = min(duration, start_sec + input_seconds + margin)
    start = int(round(read_start_sec * orig_sfreq))
    stop = min(raw.n_times, int(round(read_stop_sec * orig_sfreq)))

    data_v = raw.get_data(picks=picks, start=start, stop=stop)
    data_uv = (data_v * 1e6).astype(np.float32, copy=False)

    if bandpass is not None:
        low, high = float(bandpass[0]), float(bandpass[1])
        # Very short recordings can be too short for filtfilt padding. The NMT-4K
        # recordings are generally much longer; this fallback keeps edge cases usable.
        if data_uv.shape[-1] >= int(orig_sfreq * 3):
            data_uv = _bandpass(data_uv, orig_sfreq, low, high)

    data_uv = _resample(data_uv, orig_sfreq, target_sfreq)

    # Trim the filtering margin in resampled coordinates.
    trim_left = int(round((start_sec - read_start_sec) * target_sfreq))
    desired = int(round(input_seconds * target_sfreq))
    data_uv = data_uv[:, trim_left : trim_left + desired]

    if clip_uv is not None:
        data_uv = np.clip(data_uv, -float(clip_uv), float(clip_uv))

    if mode == "paper22":
        # Re-reference the selected 21 electrodes into the 22 bipolar derivations
        # explicitly listed in the SCNet paper.
        idx = {c: i for i, c in enumerate(required)}
        data_uv = np.stack(
            [data_uv[idx[a]] - data_uv[idx[b]] for a, b in BIPOLAR_22], axis=0
        ).astype(np.float32, copy=False)

    available_samples = int(data_uv.shape[-1])
    pad_samples = max(0, desired - available_samples)
    if available_samples < desired:
        data_uv = np.pad(data_uv, ((0, 0), (0, pad_samples)), mode="constant")
    elif available_samples > desired:
        data_uv = data_uv[:, :desired]

    out_dtype = np.float16 if cache_dtype == "float16" else np.float32
    data_uv = data_uv.astype(out_dtype, copy=False)

    info = {
        "edf_path": str(edf_path),
        "mode": mode,
        "orig_sfreq": orig_sfreq,
        "target_sfreq": target_sfreq,
        "duration_sec": duration,
        "window_start_sec": start_sec,
        "input_seconds": input_seconds,
        "available_samples_before_pad": available_samples,
        "pad_samples": pad_samples,
        "output_channels": int(data_uv.shape[0]),
        "output_samples": int(data_uv.shape[1]),
        "cache_dtype": str(data_uv.dtype),
    }
    return data_uv, info


def atomic_save_npy(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
    tmp.replace(path)
