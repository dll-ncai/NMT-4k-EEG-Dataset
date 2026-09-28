from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_project_path(value: str | Path, config_path: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (Path(config_path).resolve().parent / p).resolve()


def dataset_root(cfg: dict) -> Path:
    return Path(cfg["paths"]["dataset_root"])


def cache_root(cfg: dict, config_path: str | Path) -> Path:
    return resolve_project_path(cfg["paths"]["cache_dir"], config_path)


def output_root(cfg: dict, config_path: str | Path) -> Path:
    return resolve_project_path(cfg["paths"]["output_dir"], config_path)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolution choices where possible.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocessing_hash(cfg: dict) -> str:
    payload = {
        "channels": cfg["data"]["channels"],
        "preprocessing": cfg["preprocessing"],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def read_metadata(cfg: dict) -> pd.DataFrame:
    root = dataset_root(cfg)
    rel = cfg["data"]["metadata_relative_path"]
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {"file_name", "split", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"recordings.tsv is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    df["file_name"] = df["file_name"].astype(str).str.strip()

    valid_splits = {"train", "evaluation"}
    bad_splits = sorted(set(df["split"]) - valid_splits)
    if bad_splits:
        raise ValueError(f"Unexpected split values: {bad_splits}")

    valid_labels = {"Normal", "Abnormal"}
    bad_labels = sorted(set(df["label"]) - valid_labels)
    if bad_labels:
        raise ValueError(f"Unexpected label values: {bad_labels}")

    if df["file_name"].duplicated().any():
        dup = df.loc[df["file_name"].duplicated(), "file_name"].head().tolist()
        raise ValueError(f"Duplicate file_name values found, e.g. {dup}")

    return df


def edf_path_for_row(row: pd.Series, cfg: dict) -> Path:
    root = dataset_root(cfg)
    fname = str(row["file_name"])
    if not fname.lower().endswith(".edf"):
        fname += ".edf"
    return root / str(row["split"]) / str(row["label"]).lower() / "edf" / fname


def cache_path_for_row(row: pd.Series, cfg: dict, config_path: str | Path) -> Path:
    base = cache_root(cfg, config_path) / preprocessing_hash(cfg)
    stem = Path(str(row["file_name"])).stem
    return base / str(row["split"]) / str(row["label"]).lower() / f"{stem}.npy"


def label_to_int(label: str) -> int:
    return 1 if str(label).strip().lower() == "abnormal" else 0


def make_or_load_validation_split(
    df: pd.DataFrame,
    cfg: dict,
    config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = output_root(cfg, config_path)
    out.mkdir(parents=True, exist_ok=True)
    split_file = out / "validation_split.tsv"

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    frac = float(cfg["data"]["validation_fraction"])
    seed = int(cfg["data"]["split_seed"])

    if split_file.exists():
        assignment = pd.read_csv(split_file, sep="\t")
        required = {"file_name", "subset"}
        if not required.issubset(assignment.columns):
            raise ValueError(f"{split_file} is missing columns {required}")
        merged = train_df.merge(
            assignment[["file_name", "subset"]],
            on="file_name",
            how="left",
            validate="one_to_one",
        )
        if merged["subset"].isna().any():
            raise RuntimeError(
                "Existing validation_split.tsv does not cover the current training set. "
                "Delete it only if you intentionally want to regenerate the split."
            )
        fit_df = merged[merged["subset"] == "fit"].drop(columns=["subset"]).reset_index(drop=True)
        val_df = merged[merged["subset"] == "validation"].drop(columns=["subset"]).reset_index(drop=True)
        return fit_df, val_df

    fit_df, val_df = train_test_split(
        train_df,
        test_size=frac,
        random_state=seed,
        stratify=train_df["label"],
    )
    fit_df = fit_df.sort_values("file_name").reset_index(drop=True)
    val_df = val_df.sort_values("file_name").reset_index(drop=True)

    assignment = pd.concat(
        [
            fit_df[["file_name"]].assign(subset="fit"),
            val_df[["file_name"]].assign(subset="validation"),
        ],
        ignore_index=True,
    ).sort_values("file_name")
    assignment.to_csv(split_file, sep="\t", index=False)
    return fit_df, val_df


def _canonical_channel_name(name: str) -> str:
    s = str(name).strip().upper()
    s = re.sub(r"^EEG[\s:_-]*", "", s)
    # Typical EDF suffixes: -REF, -LE, -A1, -A2, etc.
    s = re.split(r"[\s:_-]", s, maxsplit=1)[0]
    return s


def find_channel_indices(raw_channel_names: Sequence[str], target_channels: Sequence[str]) -> list[int]:
    cleaned = [_canonical_channel_name(x) for x in raw_channel_names]
    indices = []
    used = set()
    for target in target_channels:
        t = str(target).upper()
        matches = [i for i, x in enumerate(cleaned) if x == t and i not in used]
        if not matches:
            raise ValueError(
                f"Required channel {target!r} was not found. "
                f"EDF channels: {list(raw_channel_names)}"
            )
        idx = matches[0]
        used.add(idx)
        indices.append(idx)
    return indices


def load_and_preprocess_edf(edf_path: Path, cfg: dict) -> np.ndarray:
    import mne

    pp = cfg["preprocessing"]
    targets = cfg["data"]["channels"]

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    picks = find_channel_indices(raw.ch_names, targets)

    source_sfreq = float(raw.info["sfreq"])
    requested_start_sec = max(0.0, float(pp["skip_first_seconds"]))
    max_sec = float(pp["max_duration_minutes"]) * 60.0

    total_sec = raw.n_times / source_sfreq
    min_remaining = float(
        pp.get(
            "minimum_remaining_seconds",
            cfg["windowing"].get("window_seconds", 0.0),
        )
    )
    # The 2020 TUH pipeline discarded the first 60 s. NMT contains a few much
    # shorter recordings, so fall back to the beginning if applying that crop
    # would leave less than one complete analysis window.
    if total_sec - requested_start_sec >= min_remaining:
        start_sec = requested_start_sec
    else:
        start_sec = 0.0

    stop_sec = min(total_sec, start_sec + max_sec)
    start_sample = int(round(start_sec * source_sfreq))
    stop_sample = int(round(stop_sec * source_sfreq))

    data_v = raw.get_data(
        picks=picks,
        start=start_sample,
        stop=stop_sample,
    ).astype(np.float64, copy=False)

    if data_v.shape[1] < 2:
        raise ValueError(f"Too few samples after cropping: {edf_path}")

    info = mne.create_info(
        ch_names=list(targets),
        sfreq=source_sfreq,
        ch_types=["eeg"] * len(targets),
    )
    segment = mne.io.RawArray(data_v, info, verbose="ERROR")

    segment.filter(
        l_freq=float(pp["l_freq"]),
        h_freq=float(pp["h_freq"]),
        method="fir",
        phase="zero",
        verbose="ERROR",
    )

    target_sfreq = float(pp["target_sfreq"])
    if not math.isclose(source_sfreq, target_sfreq, rel_tol=0, abs_tol=1e-6):
        segment.resample(target_sfreq, npad="auto", verbose="ERROR")

    x = segment.get_data().astype(np.float32, copy=False) * 1e6
    clip_uv = float(pp["clip_microvolts"])
    np.clip(x, -clip_uv, clip_uv, out=x)

    dtype_name = str(pp.get("cache_dtype", "float32")).lower()
    if dtype_name == "float16":
        x = x.astype(np.float16)
    elif dtype_name == "float32":
        x = x.astype(np.float32)
    else:
        raise ValueError("cache_dtype must be 'float16' or 'float32'")
    return x


def window_starts(
    n_samples: int,
    window_samples: int,
    stride_samples: int,
) -> list[int]:
    if n_samples <= window_samples:
        return [0]
    starts = list(range(0, n_samples - window_samples + 1, stride_samples))
    last = n_samples - window_samples
    if starts[-1] != last:
        starts.append(last)
    return starts


def pad_window(x: np.ndarray, window_samples: int, mode: str = "reflect") -> np.ndarray:
    if x.shape[-1] >= window_samples:
        return x[..., :window_samples]
    missing = window_samples - x.shape[-1]
    if x.shape[-1] <= 1 or mode == "constant":
        return np.pad(x, ((0, 0), (0, missing)), mode="constant")
    if mode not in {"reflect", "edge"}:
        raise ValueError(f"Unsupported padding mode: {mode}")
    # np.pad reflect cannot pad by arbitrarily many samples in one go if the
    # dimension is tiny, so repeat in safe chunks.
    out = x
    while out.shape[-1] < window_samples:
        remaining = window_samples - out.shape[-1]
        max_pad = max(1, out.shape[-1] - 1)
        pad_now = min(remaining, max_pad)
        out = np.pad(out, ((0, 0), (0, pad_now)), mode=mode)
    return out[..., :window_samples]


class CachedWindowDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict, config_path: str | Path):
        self.df = df.reset_index(drop=True).copy()
        self.cfg = cfg
        self.config_path = config_path
        sfreq = float(cfg["preprocessing"]["target_sfreq"])
        self.window_samples = int(round(float(cfg["windowing"]["window_seconds"]) * sfreq))
        self.stride_samples = int(cfg["windowing"]["stride_samples"])
        self.padding = str(cfg["windowing"].get("short_recording_padding", "reflect"))

        self.records = []
        self.index: list[tuple[int, int]] = []
        missing = []

        for record_idx, row in self.df.iterrows():
            p = cache_path_for_row(row, cfg, config_path)
            if not p.exists():
                missing.append(str(p))
                continue
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[0] != len(cfg["data"]["channels"]):
                raise ValueError(f"Unexpected cached shape {arr.shape} for {p}")
            n_samples = int(arr.shape[1])
            record_obj = {
                "path": p,
                "label": label_to_int(row["label"]),
                "file_name": row["file_name"],
                "n_samples": n_samples,
            }
            actual_idx = len(self.records)
            self.records.append(record_obj)
            for start in window_starts(n_samples, self.window_samples, self.stride_samples):
                self.index.append((actual_idx, start))

        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} required cache files are missing. "
                f"Run 03_build_cache.py first. Examples:\n{preview}"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        record_idx, start = self.index[idx]
        rec = self.records[record_idx]
        arr = np.load(rec["path"], mmap_mode="r")
        end = start + self.window_samples
        x = np.asarray(arr[:, start:end], dtype=np.float32)
        if x.shape[-1] < self.window_samples:
            x = pad_window(x, self.window_samples, self.padding)
        x = np.ascontiguousarray(x, dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(rec["label"], dtype=torch.long)


def build_model(cfg: dict) -> torch.nn.Module:
    mc = cfg["model"]
    variant = str(mc.get("variant", "paper_dense")).lower()
    common = dict(
        n_chans=len(cfg["data"]["channels"]),
        n_outputs=2,
        n_blocks=int(mc["n_blocks"]),
        n_filters=int(mc["n_filters"]),
        kernel_size=int(mc["kernel_size"]),
        drop_prob=float(mc["drop_prob"]),
    )

    if variant == "paper_dense":
        # The 2020 implementation trained on dense TCN outputs: a 6000-sample
        # crop produces 5070 predictions with the published 931-sample
        # receptive field. Modern Braindecode contains that TCN directly.
        from braindecode.models.tcn import TCN
        return TCN(**common)

    if variant == "pooled_modern":
        # Current convenience wrapper: same base TCN followed by temporal
        # adaptive-average pooling to one logit vector per input window.
        from braindecode.models import BDTCN
        sfreq = float(cfg["preprocessing"]["target_sfreq"])
        n_times = int(round(float(cfg["windowing"]["window_seconds"]) * sfreq))
        return BDTCN(n_times=n_times, **common)

    raise ValueError(
        f"Unknown model.variant={variant!r}. Use 'paper_dense' or 'pooled_modern'."
    )


def compute_metrics(
    y_true: np.ndarray,
    probs_abnormal: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs_abnormal, dtype=float)
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": float(roc_auc_score(y_true, probs)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold_f1(y_true: np.ndarray, probs_abnormal: np.ndarray) -> tuple[float, dict]:
    # Search validation probabilities plus useful boundaries. This uses no
    # evaluation labels.
    probs = np.asarray(probs_abnormal, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probs)))
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, probs, best_threshold)
    best_key = (best_metrics["f1"], best_metrics["sensitivity"], -abs(best_threshold - 0.5))

    for threshold in candidates:
        metrics = compute_metrics(y_true, probs, float(threshold))
        key = (metrics["f1"], metrics["sensitivity"], -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


@torch.no_grad()
def predict_recordings(
    model: torch.nn.Module,
    df: pd.DataFrame,
    cfg: dict,
    config_path: str | Path,
    device: torch.device,
    desc: str,
) -> pd.DataFrame:
    from tqdm import tqdm

    model.eval()
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    window_samples = int(round(float(cfg["windowing"]["window_seconds"]) * sfreq))
    stride_samples = int(cfg["windowing"]["stride_samples"])
    padding = str(cfg["windowing"].get("short_recording_padding", "reflect"))
    batch_size = int(cfg["evaluation"]["prediction_batch_size"])
    use_amp = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, unit="recording"):
        p = cache_path_for_row(row, cfg, config_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing cache: {p}")
        arr = np.load(p, mmap_mode="r")
        starts = window_starts(arr.shape[1], window_samples, stride_samples)

        probs = []
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i : i + batch_size]
            xs = []
            for start in batch_starts:
                x = np.asarray(arr[:, start : start + window_samples], dtype=np.float32)
                if x.shape[-1] < window_samples:
                    x = pad_window(x, window_samples, padding)
                xs.append(np.ascontiguousarray(x))
            xb = torch.from_numpy(np.stack(xs)).to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
            else:
                logits = model(xb)

            probs_all = torch.softmax(logits.float(), dim=1)
            if probs_all.ndim == 3:
                # B x classes x dense-time -> one probability per input window.
                pb = probs_all[:, 1, :].mean(dim=-1)
            elif probs_all.ndim == 2:
                pb = probs_all[:, 1]
            else:
                raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
            probs.extend(pb.detach().cpu().numpy().tolist())

        rows.append(
            {
                "file_name": row["file_name"],
                "label": row["label"],
                "y_true": label_to_int(row["label"]),
                "n_windows": len(starts),
                "prob_abnormal": float(np.mean(probs)),
            }
        )
    return pd.DataFrame(rows)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
