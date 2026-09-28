from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch.utils.data import DataLoader

from src.config import load_config, project_path
from src.data import CachedEEGDataset
from src.model import build_model
from src.train_utils import compute_metrics, evaluate_loader, find_best_threshold, get_device, save_json, seed_everything


def make_loader(dataset, batch_size, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="evaluation", choices=["validation", "evaluation"])
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["data"]["seed"]))

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    manifest = pd.read_csv(project_path(cfg, cfg["data"]["manifest_file"]))
    cache_dir = Path(cfg["data"]["cache_dir"])
    ev_cfg = cfg["evaluation"]
    dataset = CachedEEGDataset(manifest, cache_dir, role=args.split, include_reverse_augmentation=False)
    loader = make_loader(
        dataset,
        batch_size=int(ev_cfg["batch_size"]),
        num_workers=int(ev_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    run_dir = project_path(cfg, cfg["training"]["run_dir"])
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {checkpoint}")

    amp_enabled = bool(cfg["training"].get("amp", True) and device.type == "cuda")

    threshold_mode = str(ev_cfg.get("threshold_mode", "fixed")).lower()
    if threshold_mode == "fixed":
        threshold = float(ev_cfg.get("threshold", 0.5))
    else:
        # Threshold is selected on validation only, never on the held-out evaluation set.
        val_ds = CachedEEGDataset(manifest, cache_dir, role="validation", include_reverse_augmentation=False)
        val_loader = make_loader(
            val_ds,
            batch_size=int(ev_cfg["batch_size"]),
            num_workers=int(ev_cfg["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        val_out = evaluate_loader(model, val_loader, device, amp_enabled, desc="Threshold selection")
        threshold = find_best_threshold(val_out["y_true"], val_out["prob_abnormal"], threshold_mode)
        print(f"Selected {threshold_mode} threshold on validation set: {threshold:.3f}")

    out = evaluate_loader(model, loader, device, amp_enabled, desc=f"Evaluate {args.split}")
    metrics = compute_metrics(out["y_true"], out["prob_abnormal"], threshold=threshold)
    metrics["loss"] = float(out["loss"])
    metrics["n"] = int(len(out["y_true"]))
    metrics["split"] = args.split
    metrics["checkpoint"] = str(checkpoint)

    pred = (out["prob_abnormal"] >= threshold).astype(np.int64)
    pred_df = pd.DataFrame({
        "sample_id": out["sample_id"],
        "true_y": out["y_true"],
        "true_label": ["abnormal" if y == 1 else "normal" for y in out["y_true"]],
        "prob_abnormal": out["prob_abnormal"],
        "pred_y": pred,
        "pred_label": ["abnormal" if y == 1 else "normal" for y in pred],
    })

    out_dir = run_dir / f"{args.split}_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    save_json(metrics, out_dir / "metrics.json")

    # Confusion matrix.
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        out["y_true"], pred,
        labels=[0, 1], display_labels=["Normal", "Abnormal"],
        values_format="d", ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # ROC curve.
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(out["y_true"], out["prob_abnormal"], ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close(fig)

    print("\nFinal metrics")
    print("-------------")
    print(f"N           = {metrics['n']}")
    print(f"Accuracy    = {metrics['accuracy']*100:.2f}%")
    print(f"F1 score    = {metrics['f1']*100:.2f}%")
    print(f"Sensitivity = {metrics['sensitivity']*100:.2f}%")
    print(f"Specificity = {metrics['specificity']*100:.2f}%")
    print(f"AUROC       = {metrics['auroc']*100:.2f}%")
    print(f"Threshold   = {metrics['threshold']:.3f}")
    print(f"TN/FP/FN/TP = {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}")
    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
