from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


SCALP_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
    "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2",
]

# 20-channel TCP montage shown in the WaveNet-LSTM paper.
TCP_PAIRS: List[Tuple[str, str]] = [
    ("FP2", "F8"), ("T4", "T6"), ("T6", "O2"), ("T4", "C4"), ("CZ", "C4"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"), ("F8", "T4"),
    ("FP1", "F7"), ("T3", "T5"), ("T5", "O1"), ("T3", "C3"), ("C3", "CZ"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"), ("F7", "T3"),
]
TCP_NAMES = [f"{a}-{b}" for a, b in TCP_PAIRS]


def normalize_channel_name(name: str) -> str:
    s = name.upper().strip()
    s = re.sub(r"^EEG\s*", "", s)
    s = s.replace(" ", "")
    for suffix in ("-REF", "-LE", "-AVG"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Some EDF exporters retain the acquisition reference in the label.
    # Only remove it when the remaining token is a recognized scalp channel.
    for suffix in ("-A1", "-A2"):
        if s.endswith(suffix) and s[: -len(suffix)] in SCALP_CHANNELS:
            s = s[: -len(suffix)]
    return s


def resolve_scalp_channels(raw: mne.io.BaseRaw) -> Tuple[List[str], Dict[str, str]]:
    normalized_to_raw: Dict[str, str] = {}
    for raw_name in raw.ch_names:
        key = normalize_channel_name(raw_name)
        if key not in normalized_to_raw:
            normalized_to_raw[key] = raw_name

    missing = [ch for ch in SCALP_CHANNELS if ch not in normalized_to_raw]
    if missing:
        raise ValueError(
            f"Missing required scalp channels: {missing}. EDF channels: {raw.ch_names}"
        )
    picks = [normalized_to_raw[ch] for ch in SCALP_CHANNELS]
    return picks, normalized_to_raw


def bandpass_filter(data: np.ndarray, sfreq: float, low: float, high: float, order: int) -> np.ndarray:
    nyq = sfreq / 2.0
    high = min(high, nyq - 0.5)
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid bandpass [{low}, {high}] for sfreq={sfreq}")
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def resample_data(data: np.ndarray, orig_sfreq: float, target_sfreq: float) -> np.ndarray:
    if math.isclose(orig_sfreq, target_sfreq, rel_tol=0, abs_tol=1e-6):
        return data
    ratio = Fraction(target_sfreq / orig_sfreq).limit_denominator(1000)
    return resample_poly(data, up=ratio.numerator, down=ratio.denominator, axis=-1)


def make_tcp(scalp_data: np.ndarray) -> np.ndarray:
    """Convert ordered 19-channel scalp data [19, T] to TCP [20, T]."""
    index = {ch: i for i, ch in enumerate(SCALP_CHANNELS)}
    out = []
    for a, b in TCP_PAIRS:
        out.append(scalp_data[index[a]] - scalp_data[index[b]])
    return np.asarray(out, dtype=np.float32)


def zscore_segment(seg_ct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score. Input/output shape [C, T]."""
    mean = seg_ct.mean(axis=1, keepdims=True)
    std = seg_ct.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return ((seg_ct - mean) / std).astype(np.float32, copy=False)


def fixed_length_segment(
    tcp_ct: np.ndarray,
    start_sample: int,
    length_samples: int,
    normalization: str,
) -> Tuple[np.ndarray, int]:
    """Return [T, C] segment and number of real (unpadded) samples."""
    available = max(0, min(length_samples, tcp_ct.shape[1] - start_sample))
    seg = tcp_ct[:, start_sample : start_sample + available]

    if available == 0:
        seg = np.zeros((tcp_ct.shape[0], 1), dtype=np.float32)

    if normalization.lower() == "zscore":
        seg = zscore_segment(seg)
    elif normalization.lower() in ("none", "off"):
        seg = seg.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if seg.shape[1] < length_samples:
        pad = np.zeros((seg.shape[0], length_samples - seg.shape[1]), dtype=np.float32)
        seg = np.concatenate([seg, pad], axis=1)
    elif seg.shape[1] > length_samples:
        seg = seg[:, :length_samples]

    return np.ascontiguousarray(seg.T, dtype=np.float32), available


def preprocess_edf(
    edf_path: str | Path,
    filter_low_hz: float,
    filter_high_hz: float,
    filter_order: int,
    target_sfreq_hz: float,
    segment_seconds: float,
    normalization: str,
    need_second_segment: bool,
) -> Dict[str, object]:
    """Load only the beginning of an EDF and create first/reversed-second 60 s segments."""
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    picks, _ = resolve_scalp_channels(raw)
    sfreq = float(raw.info["sfreq"])

    need_seconds = segment_seconds * (2 if need_second_segment else 1)
    stop = min(raw.n_times, int(math.ceil(need_seconds * sfreq)))
    data = raw.get_data(picks=picks, start=0, stop=stop, verbose="ERROR").astype(np.float64, copy=False)
    raw.close()

    data = bandpass_filter(data, sfreq, filter_low_hz, filter_high_hz, filter_order)
    data = resample_data(data, sfreq, target_sfreq_hz)
    tcp = make_tcp(data)

    n = int(round(segment_seconds * target_sfreq_hz))
    first, first_real = fixed_length_segment(tcp, 0, n, normalization)

    second = None
    second_real = 0
    if need_second_segment:
        second, second_real = fixed_length_segment(tcp, n, n, normalization)
        # The paper's augmentation reverses the second 60-second segment in time.
        second = np.ascontiguousarray(second[::-1], dtype=np.float32)

    return {
        "first": first,
        "second_reversed": second,
        "first_real_samples": int(first_real),
        "second_real_samples": int(second_real),
        "orig_sfreq": sfreq,
        "target_sfreq": float(target_sfreq_hz),
        "input_shape": tuple(first.shape),
    }
