from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is useful for reproducibility; benchmark may be slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, probs, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "auroc": float(auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def evaluate_loader(model, loader, device, amp: bool, desc: str = "Evaluate"):
    model.eval()
    ys, probs, ids, labels_text = [], [], [], []
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    enabled_amp = bool(amp and device.type == "cuda")

    with torch.no_grad():
        for x, y, sample_ids, label_text in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled_amp):
                logits = model(x)
                loss = criterion(logits, y)
            p = torch.softmax(logits.float(), dim=1)[:, 1]
            losses.append(float(loss.item()) * x.size(0))
            ys.extend(y.detach().cpu().numpy().tolist())
            probs.extend(p.detach().cpu().numpy().tolist())
            ids.extend(list(sample_ids))
            labels_text.extend(list(label_text))

    return {
        "loss": sum(losses) / max(1, len(ys)),
        "y_true": np.asarray(ys, dtype=np.int64),
        "prob_abnormal": np.asarray(probs, dtype=np.float64),
        "sample_id": ids,
        "label_text": labels_text,
    }


def find_best_threshold(y_true, probs, mode: str) -> float:
    mode = mode.lower()
    if mode == "fixed":
        raise ValueError("fixed threshold should be supplied directly")
    candidates = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -np.inf
    for t in candidates:
        m = compute_metrics(y_true, probs, float(t))
        if mode == "f1":
            score = m["f1"]
        elif mode == "youden":
            score = m["sensitivity"] + m["specificity"] - 1.0
        else:
            raise ValueError(f"Unknown threshold_mode={mode}")
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def atomic_torch_save(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
