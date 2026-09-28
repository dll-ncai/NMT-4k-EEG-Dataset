from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import load_config, resolve_path, save_json, seed_everything
from src.data import CachedEEGDataset, make_or_load_splits
from src.metrics import binary_metrics
from src.scnet import SCNet


def make_loader(dataset, batch_size, workers):
    kwargs = dict(
        dataset=dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def pct_metrics(m: dict) -> dict:
    out = dict(m)
    for key in ("accuracy", "f1", "sensitivity", "specificity", "auroc"):
        out[key + "_percent"] = 100.0 * out[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SCNet on the held-out NMT-4K evaluation set.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Default: output_dir/checkpoints/best.pt")
    parser.add_argument("--threshold", default="auto", help="auto, 0.5, or any float in [0,1]")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base = cfg_path.parent
    cache_dir = resolve_path(base, cfg["cache_dir"])
    output_dir = resolve_path(base, cfg["output_dir"])
    train_cfg = cfg["train"]
    seed_everything(int(train_cfg.get("seed", 42)))

    manifest = pd.read_csv(cache_dir / "manifest.csv", dtype={"file_name": str})
    _, _, eval_df = make_or_load_splits(
        manifest, output_dir,
        val_fraction=float(train_cfg.get("val_fraction", 0.15)),
        seed=int(train_cfg.get("seed", 42)),
    )
    expected_eval = int((manifest["split"].str.lower() == "evaluation").sum())
    if len(eval_df) != expected_eval:
        raise RuntimeError(f"Only {len(eval_df)}/{expected_eval} evaluation recordings have status=ok in cache.")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else output_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_channels = int(ckpt.get("input_channels", 19 if cfg["preprocess"]["mode"] == "nmt4k19" else 22))
    model = SCNet(input_channels=input_channels, num_classes=2)
    model.load_state_dict(ckpt["model_state"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run check_setup.py first.")
    device = torch.device("cuda")
    model.to(device).eval()

    batch_size = int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 8)))
    workers = int(train_cfg.get("num_workers", 2))
    amp_enabled = bool(train_cfg.get("amp", True))
    loader = make_loader(CachedEEGDataset(eval_df, cache_dir), batch_size, workers)

    ys, probs, names = [], [], []
    with torch.no_grad():
        for x, y, batch_names in tqdm(loader, desc="Held-out evaluation", unit="batch"):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
            p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            ys.append(y.numpy())
            probs.append(p)
            names.extend(list(batch_names))

    y_true = np.concatenate(ys)
    prob = np.concatenate(probs)

    if str(args.threshold).lower() == "auto":
        threshold = float(ckpt.get("best_threshold", 0.5))
    else:
        threshold = float(args.threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

    metrics_selected = binary_metrics(y_true, prob, threshold)
    metrics_05 = binary_metrics(y_true, prob, 0.5)
    all_metrics = {
        "selected_threshold_metrics": pct_metrics(metrics_selected),
        "threshold_0.5_metrics": pct_metrics(metrics_05),
        "n_evaluation": int(len(y_true)),
        "positive_class": "abnormal",
        "checkpoint": str(ckpt_path),
        "best_training_epoch": int(ckpt.get("best_epoch", -1)),
    }
    save_json(all_metrics, output_dir / "evaluation_metrics.json")

    pred = (prob >= threshold).astype(int)
    predictions = pd.DataFrame({
        "file_name": names,
        "label_true": np.where(y_true == 1, "abnormal", "normal"),
        "target_true": y_true,
        "prob_abnormal": prob,
        "threshold": threshold,
        "predicted_label": np.where(pred == 1, "abnormal", "normal"),
        "correct": pred == y_true,
    })
    predictions.to_csv(output_dir / "evaluation_predictions.csv", index=False)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Abnormal"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d", cmap=None, colorbar=False)
    ax.set_title(f"SCNet NMT-4K Evaluation (threshold={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_confusion_matrix.png", dpi=200)
    plt.close(fig)

    m = metrics_selected
    print("\nSCNet held-out evaluation results")
    print(f"N           : {len(y_true)}")
    print(f"Threshold   : {threshold:.3f}")
    print(f"Accuracy    : {m['accuracy']*100:.2f}%")
    print(f"F1 score    : {m['f1']*100:.2f}%")
    print(f"Sensitivity : {m['sensitivity']*100:.2f}%")
    print(f"Specificity : {m['specificity']*100:.2f}%")
    print(f"AUROC       : {m['auroc']*100:.2f}%")
    print(f"TN/FP/FN/TP : {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}")
    print(f"\nSaved metrics:     {output_dir / 'evaluation_metrics.json'}")
    print(f"Saved predictions: {output_dir / 'evaluation_predictions.csv'}")


if __name__ == "__main__":
    main()
