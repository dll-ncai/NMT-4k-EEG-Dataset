from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ensure_output_directories, load_config, resolved_config
from .data import WindowedRecordingDataset, window_parameters
from .metrics import (
    aggregate_recording_predictions,
    binary_metrics,
    curve_values,
)
from .model import build_model
from .runtime import amp_context
from .utils import atomic_write_json, human_duration, print_heading


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _plot_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    curves = curve_values(y_true, probabilities)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].plot(curves["fpr"], curves["tpr"], color="#2463EB", linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    axes[0].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"Recording-level ROC (AUROC={metrics['auroc']:.4f})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].grid(alpha=0.2)

    prevalence = float(np.mean(y_true))
    axes[1].plot(curves["recall"], curves["precision"], color="#E4572E", linewidth=2)
    axes[1].axhline(prevalence, linestyle="--", color="#888888", linewidth=1)
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Recording-level PR (PR-AUC={metrics['pr_auc']:.4f})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].grid(alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    pd.DataFrame(
        {
            "false_positive_rate": curves["fpr"],
            "true_positive_rate": curves["tpr"],
            "threshold": curves["roc_thresholds"],
        }
    ).to_csv(output_path.with_name("roc_curve.csv"), index=False)
    # precision/recall has one more point than its threshold vector.
    pr_thresholds = np.append(curves["pr_thresholds"], np.nan)
    pd.DataFrame(
        {
            "recall": curves["recall"],
            "precision": curves["precision"],
            "threshold": pr_thresholds,
        }
    ).to_csv(output_path.with_name("precision_recall_curve.csv"), index=False)


def evaluate_pipeline(config_path: str | Path) -> int:
    config, paths = load_config(config_path)
    ensure_output_directories(paths)
    requested_device = str(config["training"].get("device", "cuda")).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run 01_verify_install.ps1 first.")

    seed = int(config["training"]["seed"])
    run_dir = paths.runs_dir / f"seed_{seed}"
    completion_path = run_dir / "training_complete.json"
    if not completion_path.exists():
        raise FileNotFoundError(
            "Training completion record is missing. Finish all development epochs "
            "before running the official evaluation."
        )
    with completion_path.open("r", encoding="utf-8") as stream:
        training_completion = json.load(stream)
    if training_completion.get("status") != "complete":
        raise RuntimeError("Training is not marked complete; evaluation is blocked.")
    required_strategy = "best_internal_validation_auroc_checkpoint_no_retraining"
    if training_completion.get("final_model_strategy") != required_strategy:
        raise RuntimeError(
            "Training was not completed with the best-validation-checkpoint-only "
            "protocol. Rerun training with the updated scripts."
        )

    selected_epoch = int(training_completion["selected_epoch"])
    checkpoint_path = Path(str(training_completion["final_checkpoint"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best validation checkpoint export not found: {checkpoint_path}. "
            "Run training first."
        )

    manifest_path = paths.audit_dir / "processed_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Processed manifest not found. Run preprocessing first."
        )
    manifest = pd.read_csv(manifest_path)
    evaluation_frame = (
        manifest[manifest["split"] == "evaluation"].copy().reset_index(drop=True)
    )
    expected_count = sum(config["dataset"]["expected_counts"]["evaluation"].values())
    if len(evaluation_frame) != expected_count:
        raise RuntimeError(
            f"Evaluation requires all {expected_count} recordings; processed manifest has {len(evaluation_frame)}."
        )

    checkpoint = _torch_load(checkpoint_path, "cpu")
    if int(checkpoint.get("epoch", -1)) != selected_epoch:
        raise RuntimeError(
            "Checkpoint epoch does not match the validation-selected best epoch."
        )
    if checkpoint.get("phase") != "development":
        raise RuntimeError(
            "Official evaluation is restricted to the saved development checkpoint "
            "with the best internal-validation AUROC."
        )
    channels = list(config["dataset"]["channels"])
    if list(checkpoint.get("channels", [])) != channels:
        raise RuntimeError("Checkpoint channel order differs from config.yaml.")

    device = torch.device(requested_device)
    model = build_model(config, n_channels=len(channels)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    window_size, stride, include_final = window_parameters(config)
    dataset = WindowedRecordingDataset(
        evaluation_frame, window_size, stride, include_final
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=bool(config["training"]["pin_memory"]),
        persistent_workers=bool(int(config["evaluation"]["num_workers"]) > 0),
        drop_last=False,
    )
    threshold = float(config["evaluation"]["threshold"])
    amp_enabled = bool(config["training"]["amp"])

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "resolved_config.json", resolved_config(config, paths)
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    recording_ids: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    print_heading("Untouched NMT-4K evaluation")
    print(f"checkpoint: {checkpoint_path}")
    print(f"selected epoch: {selected_epoch}")
    print(
        "best internal-validation AUROC: "
        f"{float(training_completion['best_validation_auroc']):.4f}"
    )
    print(f"recordings: {len(evaluation_frame)}")
    print(f"windows: {len(dataset)}")

    with torch.inference_mode():
        with tqdm(
            loader,
            total=len(loader),
            desc=f"official evaluation best E{selected_epoch:02d}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
            smoothing=0.1,
        ) as progress:
            for inputs, targets, batch_recording_ids in progress:
                inputs = inputs.to(device, non_blocking=True)
                with amp_context(device, amp_enabled):
                    logits = model(inputs)
                abnormal_probability = torch.softmax(logits.float(), dim=1)[:, 1]
                recording_ids.extend(str(value) for value in batch_recording_ids)
                labels.extend(targets.numpy().astype(int).tolist())
                probabilities.extend(abnormal_probability.cpu().numpy().tolist())

    elapsed = time.perf_counter() - start
    window_predictions = pd.DataFrame(
        {
            "recording_id": recording_ids,
            "true_label": labels,
            "abnormal_probability": probabilities,
        }
    )
    window_predictions["window_number"] = window_predictions.groupby(
        "recording_id"
    ).cumcount()
    window_predictions["predicted_label"] = (
        window_predictions["abnormal_probability"] >= threshold
    ).astype(int)
    window_metrics = binary_metrics(labels, probabilities, threshold)

    recording_predictions = aggregate_recording_predictions(
        recording_ids, labels, probabilities
    )
    recording_predictions["predicted_label"] = (
        recording_predictions["abnormal_probability"] >= threshold
    ).astype(int)
    recording_predictions["true_class"] = recording_predictions["true_label"].map(
        {0: "normal", 1: "abnormal"}
    )
    recording_predictions["predicted_class"] = recording_predictions[
        "predicted_label"
    ].map({0: "normal", 1: "abnormal"})
    recording_metrics = binary_metrics(
        recording_predictions["true_label"],
        recording_predictions["abnormal_probability"],
        threshold,
    )

    if bool(config["evaluation"]["save_window_predictions"]):
        window_predictions.to_csv(output_dir / "window_predictions.csv", index=False)
    recording_predictions.to_csv(output_dir / "recording_predictions.csv", index=False)
    atomic_write_json(output_dir / "window_metrics.json", window_metrics)
    atomic_write_json(output_dir / "recording_metrics.json", recording_metrics)
    pd.DataFrame([recording_metrics]).to_csv(
        output_dir / "recording_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            [recording_metrics["tn"], recording_metrics["fp"]],
            [recording_metrics["fn"], recording_metrics["tp"]],
        ],
        index=["true_normal", "true_abnormal"],
        columns=["predicted_normal", "predicted_abnormal"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    if bool(config["evaluation"]["plot_curves"]):
        _plot_curves(
            recording_predictions["true_label"].to_numpy(dtype=int),
            recording_predictions["abnormal_probability"].to_numpy(dtype=float),
            recording_metrics,
            output_dir / "roc_pr_curves.png",
        )

    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_role": "best_internal_validation_auroc_no_retraining",
        "selected_epoch": selected_epoch,
        "best_internal_validation_auroc": float(
            training_completion["best_validation_auroc"]
        ),
        "recordings": len(recording_predictions),
        "windows": len(window_predictions),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated() / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "primary_level": "recording",
        "recording_metrics": recording_metrics,
        "window_metrics": window_metrics,
    }
    atomic_write_json(output_dir / "evaluation_summary.json", summary)

    print("\nPrimary recording-level results")
    print(f"Accuracy    : {recording_metrics['accuracy']:.4f}")
    print(f"F1 abnormal : {recording_metrics['f1_abnormal']:.4f}")
    print(f"Sensitivity : {recording_metrics['sensitivity']:.4f}")
    print(f"Specificity : {recording_metrics['specificity']:.4f}")
    print(f"AUROC       : {recording_metrics['auroc']:.4f}")
    print(f"Balanced Acc: {recording_metrics['balanced_accuracy']:.4f}")
    print(f"PR-AUC      : {recording_metrics['pr_auc']:.4f}")
    print(
        f"Confusion   : TN={recording_metrics['tn']} FP={recording_metrics['fp']} "
        f"FN={recording_metrics['fn']} TP={recording_metrics['tp']}"
    )
    print(f"Elapsed     : {human_duration(elapsed)}")
    print(f"Outputs     : {output_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Multi-BK-Net on NMT-4K.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    try:
        raise SystemExit(evaluate_pipeline(args.config))
    except torch.cuda.OutOfMemoryError as exc:
        print(f"CUDA out of memory during evaluation: {exc}")
        print(
            "Reduce evaluation.batch_size in config.yaml, then rerun 05_evaluate.ps1."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
