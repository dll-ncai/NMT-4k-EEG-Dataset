from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


@dataclass
class BinaryMetrics:
    threshold: float
    accuracy: float
    f1: float
    sensitivity: float
    specificity: float
    auroc: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> BinaryMetrics:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return BinaryMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        auroc=float(auroc),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def find_best_f1_threshold(y_true, y_prob) -> tuple[float, BinaryMetrics]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    # Include 0.5 and probability-derived midpoints. This is exact enough for validation-sized sets.
    unique = np.unique(y_prob)
    if len(unique) <= 1:
        candidates = np.array([0.5], dtype=np.float64)
    else:
        mids = (unique[:-1] + unique[1:]) / 2.0
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique, mids)))
    best_t = 0.5
    best_m = compute_binary_metrics(y_true, y_prob, best_t)
    for t in candidates:
        m = compute_binary_metrics(y_true, y_prob, float(t))
        if (m.f1 > best_m.f1 + 1e-12) or (
            abs(m.f1 - best_m.f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)
        ):
            best_t, best_m = float(t), m
    return best_t, best_m
