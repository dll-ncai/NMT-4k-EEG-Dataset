from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from nmt_deep4.checkpoint import load_checkpoint
from nmt_deep4.common import (
    device_from_config,
    ensure_dirs,
    get_paths,
    load_config,
    save_json,
    set_global_seed,
)
from nmt_deep4.data import FixedWindowDataset, load_channel_stats, load_manifest
from nmt_deep4.eval_utils import predict_recordings
from nmt_deep4.metrics import compute_binary_metrics, find_best_f1_threshold
from nmt_deep4.model import Deep4Net


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def make_loader(
    records: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    n_times: int,
    stride_samples: int,
    max_windows: int | None,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Use exactly the same exhaustive/sliding-window scheme for validation and evaluation."""
    ds = FixedWindowDataset(
        records,
        n_times=n_times,
        mean=mean,
        std=std,
        windows_per_recording=None,
        stride_samples=stride_samples,
        max_windows_per_recording=max_windows,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )


def check_predictions(pred: pd.DataFrame, split_name: str) -> None:
    if len(pred) == 0:
        raise RuntimeError(f"No recording-level predictions were produced for {split_name}.")
    p = pred["prob_abnormal"].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all():
        bad = int((~np.isfinite(p)).sum())
        raise FloatingPointError(
            f"{split_name} produced {bad} NaN/Inf recording probabilities. "
            "Do not use these metrics."
        )
    if np.any((p < 0.0) | (p > 1.0)):
        raise FloatingPointError(
            f"{split_name} contains probabilities outside [0, 1]."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", choices=["evaluation", "val"], default="evaluation")
    ap.add_argument(
        "--recalibrate-threshold",
        action="store_true",
        help="Force recomputation of the exhaustive-validation F1 threshold.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    set_global_seed(int(cfg.get("seed", 42)))
    device = device_from_config(cfg)

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df = load_manifest(paths["manifest"])

    if not paths["stats"].exists():
        raise FileNotFoundError(
            f"Training normalization stats not found: {paths['stats']}. "
            "Run 02_preprocess_cache.py first."
        )

    mean, std = load_channel_stats(paths["stats"])
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics contain NaN/Inf.")
    if np.any(std <= 0):
        raise ValueError("Normalization statistics contain non-positive standard deviations.")

    sfreq = float(cfg["preprocessing"].get("sfreq", 100.0))
    tcfg = cfg["training"]
    ecfg = cfg.get("evaluation", {})

    n_times = int(round(float(tcfg.get("window_seconds", 6.0)) * sfreq))
    stride_seconds = float(
        ecfg.get("stride_seconds", tcfg.get("window_seconds", 6.0))
    )
    stride_samples = int(round(stride_seconds * sfreq))

    max_windows = ecfg.get("max_windows_per_recording", None)
    max_windows = None if max_windows is None else int(max_windows)

    batch_size = int(tcfg.get("eval_batch_size", 128))
    num_workers = int(tcfg.get("num_workers", 2))

    # ------------------------------------------------------------------
    # IMPORTANT: the safe training run automatically disabled fp16 AMP.
    # Keep evaluation in FP32 by default as well. If you later explicitly
    # want AMP for inference, add:
    #
    # evaluation:
    #   amp: true
    #
    # to config.yaml. For the current run, leave it OFF.
    # ------------------------------------------------------------------
    use_amp = bool(ecfg.get("amp", False) and device.type == "cuda")
    print(f"Evaluation automatic mixed precision (AMP): {'ON' if use_amp else 'OFF'}")
    print(
        f"Windowing: {float(tcfg.get('window_seconds', 6.0)):.2f} s windows, "
        f"{stride_seconds:.2f} s stride, "
        f"max_windows_per_recording={max_windows}"
    )

    model = Deep4Net(
        n_chans=19,
        n_times=n_times,
        drop_prob=float(tcfg.get("dropout", 0.5)),
    ).to(device)

    best_path = paths["checkpoint_dir"] / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_path}")

    load_checkpoint(best_path, model, map_location=device)
    print("Loaded checkpoint:", best_path)

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------
    # The training script selected threshold.json using 16 uniformly-spaced
    # validation windows per recording. Final evaluation uses exhaustive
    # sliding/non-overlapping windows. A threshold should be calibrated with
    # the SAME aggregation rule used at test time.
    #
    # Therefore this script recalibrates the threshold on the validation
    # partition using exactly the same exhaustive windowing as evaluation.
    # The held-out evaluation partition is never used for threshold choice.
    # ------------------------------------------------------------------
    threshold_exhaustive_path = paths["artifacts_dir"] / "threshold_exhaustive.json"
    val_pred_path = paths["artifacts_dir"] / "validation_exhaustive_predictions.csv"
    val_metrics_path = paths["artifacts_dir"] / "validation_exhaustive_metrics.json"

    if threshold_exhaustive_path.exists() and not args.recalibrate_threshold:
        threshold_info = json.loads(
            threshold_exhaustive_path.read_text(encoding="utf-8")
        )
        threshold = float(threshold_info["threshold"])
        print(
            f"Using existing exhaustive-validation threshold: {threshold:.6f}"
        )
    else:
        val_records = df[df["role"] == "val"].reset_index(drop=True)
        if len(val_records) == 0:
            raise ValueError("No validation records found in manifest.")

        print(
            f"\nCalibrating threshold on validation only "
            f"({len(val_records)} recordings) using the SAME window aggregation "
            f"as final evaluation..."
        )

        val_loader = make_loader(
            val_records,
            mean,
            std,
            n_times,
            stride_samples,
            max_windows,
            batch_size,
            num_workers,
        )

        val_pred = predict_recordings(
            model,
            val_loader,
            val_records,
            device,
            amp=use_amp,
            desc="Validation exhaustive",
        )
        check_predictions(val_pred, "validation")

        threshold, val_best = find_best_f1_threshold(
            val_pred["target"], val_pred["prob_abnormal"]
        )
        val_m05 = compute_binary_metrics(
            val_pred["target"], val_pred["prob_abnormal"], threshold=0.5
        )

        val_pred["pred_selected"] = (
            val_pred["prob_abnormal"] >= threshold
        ).astype(int)
        val_pred["pred_0.5"] = (
            val_pred["prob_abnormal"] >= 0.5
        ).astype(int)
        val_pred.to_csv(val_pred_path, index=False)

        save_json(
            {
                "selected_on": "validation",
                "criterion": "maximum F1",
                "aggregation": "mean probability across exhaustive/sliding windows per recording",
                "window_seconds": float(tcfg.get("window_seconds", 6.0)),
                "stride_seconds": stride_seconds,
                "max_windows_per_recording": max_windows,
                "amp": use_amp,
                "threshold": float(threshold),
                "metrics_at_selected_threshold": val_best.to_dict(),
                "metrics_at_0.5": val_m05.to_dict(),
            },
            threshold_exhaustive_path,
        )

        save_json(
            {
                "positive_class": "Abnormal",
                "aggregation": "mean probability across exhaustive/sliding windows per recording",
                "window_seconds": float(tcfg.get("window_seconds", 6.0)),
                "stride_seconds": stride_seconds,
                "max_windows_per_recording": max_windows,
                "amp": use_amp,
                "selected_threshold": val_best.to_dict(),
                "threshold_0.5": val_m05.to_dict(),
            },
            val_metrics_path,
        )

        print("\nExhaustive-validation threshold calibration")
        print(f"Threshold     : {threshold:.6f}")
        print(f"Accuracy      : {fmt_pct(val_best.accuracy)}")
        print(f"F1            : {fmt_pct(val_best.f1)}")
        print(f"Sensitivity   : {fmt_pct(val_best.sensitivity)}")
        print(f"Specificity   : {fmt_pct(val_best.specificity)}")
        print(f"AUROC         : {fmt_pct(val_best.auroc)}")
        print(
            f"Confusion     : TN={val_best.tn}, FP={val_best.fp}, "
            f"FN={val_best.fn}, TP={val_best.tp}"
        )
        print("Saved threshold:", threshold_exhaustive_path)

    if args.split == "val":
        print(
            "\nValidation-only run complete. "
            "No held-out evaluation recordings were scored."
        )
        return

    # ------------------------------------------------------------------
    # FINAL held-out evaluation
    # ------------------------------------------------------------------
    records = df[df["role"] == "evaluation"].reset_index(drop=True)
    if len(records) == 0:
        raise ValueError("No evaluation records found in manifest.")

    print(
        f"\nEvaluating held-out partition: {len(records)} recordings. "
        "Threshold is frozen from validation."
    )

    loader = make_loader(
        records,
        mean,
        std,
        n_times,
        stride_samples,
        max_windows,
        batch_size,
        num_workers,
    )

    pred = predict_recordings(
        model,
        loader,
        records,
        device,
        amp=use_amp,
        desc="Evaluating evaluation",
    )
    check_predictions(pred, "evaluation")

    pred["pred_selected"] = (
        pred["prob_abnormal"] >= threshold
    ).astype(int)
    pred["pred_0.5"] = (
        pred["prob_abnormal"] >= 0.5
    ).astype(int)

    m_selected = compute_binary_metrics(
        pred["target"], pred["prob_abnormal"], threshold
    )
    m_05 = compute_binary_metrics(
        pred["target"], pred["prob_abnormal"], 0.5
    )

    pred_path = paths["artifacts_dir"] / "evaluation_predictions.csv"
    metrics_path = paths["artifacts_dir"] / "evaluation_metrics.json"

    pred.to_csv(pred_path, index=False)
    save_json(
        {
            "positive_class": "Abnormal",
            "checkpoint": str(best_path),
            "threshold_source": "validation_exhaustive_same_aggregation",
            "aggregation": "mean probability across exhaustive/sliding windows per recording",
            "window_seconds": float(tcfg.get("window_seconds", 6.0)),
            "stride_seconds": stride_seconds,
            "max_windows_per_recording": max_windows,
            "amp": use_amp,
            "selected_threshold": m_selected.to_dict(),
            "threshold_0.5": m_05.to_dict(),
        },
        metrics_path,
    )

    print("\nFINAL recording-level results (positive class = Abnormal)")
    print(f"Threshold     : {threshold:.6f} (selected on validation only)")
    print(f"Accuracy      : {fmt_pct(m_selected.accuracy)}")
    print(f"F1            : {fmt_pct(m_selected.f1)}")
    print(f"Sensitivity   : {fmt_pct(m_selected.sensitivity)}")
    print(f"Specificity   : {fmt_pct(m_selected.specificity)}")
    print(f"AUROC         : {fmt_pct(m_selected.auroc)}")
    print(
        f"Confusion     : TN={m_selected.tn}, FP={m_selected.fp}, "
        f"FN={m_selected.fn}, TP={m_selected.tp}"
    )
    print("\nPredictions:", pred_path)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
