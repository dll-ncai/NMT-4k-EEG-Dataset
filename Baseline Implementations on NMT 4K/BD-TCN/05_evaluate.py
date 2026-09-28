from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from bdtcn_nmt_utils import (
    build_model,
    compute_metrics,
    load_config,
    output_root,
    predict_recordings,
    read_metadata,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Default: <output_dir>/checkpoints/best.pt",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out = output_root(cfg, args.config)
    checkpoint = (
        __import__("pathlib").Path(args.checkpoint)
        if args.checkpoint
        else out / "checkpoints" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\nRun 04_train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if "validation_threshold" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain a validation-selected threshold. "
            "Do not select a threshold on the evaluation set."
        )
    threshold = float(ckpt["validation_threshold"])
    print(f"Validation-selected threshold: {threshold:.6f}")

    df = read_metadata(cfg)
    eval_df = df[df["split"] == "evaluation"].copy().sort_values("file_name").reset_index(drop=True)
    print(f"Evaluation recordings: {len(eval_df):,}")
    print(eval_df["label"].value_counts().to_string())

    pred = predict_recordings(
        model,
        eval_df,
        cfg,
        args.config,
        device,
        desc="Held-out evaluation",
    )
    metrics = compute_metrics(
        pred["y_true"].to_numpy(),
        pred["prob_abnormal"].to_numpy(),
        threshold,
    )
    pred["y_pred"] = (pred["prob_abnormal"] >= threshold).astype(int)
    pred["predicted_label"] = pred["y_pred"].map({0: "Normal", 1: "Abnormal"})

    predictions_path = out / "evaluation_predictions.tsv"
    metrics_path = out / "evaluation_metrics.json"
    pred.to_csv(predictions_path, sep="\t", index=False)
    save_json(metrics, metrics_path)

    print("\nFinal held-out metrics")
    print("=" * 42)
    print(f"Accuracy    : {metrics['accuracy'] * 100:.2f}%")
    print(f"F1          : {metrics['f1'] * 100:.2f}%")
    print(f"Sensitivity : {metrics['sensitivity'] * 100:.2f}%")
    print(f"Specificity : {metrics['specificity'] * 100:.2f}%")
    print(f"AUROC       : {metrics['auroc'] * 100:.2f}%")
    print(f"Threshold   : {metrics['threshold']:.6f}")
    print(f"Confusion   : TN={metrics['tn']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TP={metrics['tp']}")
    print("=" * 42)
    print(f"Predictions : {predictions_path}")
    print(f"Metrics     : {metrics_path}")


if __name__ == "__main__":
    main()
