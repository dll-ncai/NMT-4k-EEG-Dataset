from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def aggregate_recording_predictions(
    recording_ids: Iterable[str],
    labels: Iterable[int],
    abnormal_probabilities: Iterable[float],
) -> pd.DataFrame:
    windows = pd.DataFrame(
        {
            "recording_id": list(recording_ids),
            "label": np.asarray(list(labels), dtype=int),
            "abnormal_probability": np.asarray(
                list(abnormal_probabilities), dtype=float
            ),
        }
    )
    if windows.empty:
        raise ValueError("No predictions were supplied for aggregation.")
    label_counts = windows.groupby("recording_id")["label"].nunique()
    inconsistent = label_counts[label_counts != 1]
    if len(inconsistent):
        raise ValueError(
            f"Recordings contain inconsistent labels: {inconsistent.index.tolist()}"
        )

    grouped = (
        windows.groupby("recording_id", sort=True)
        .agg(
            true_label=("label", "first"),
            abnormal_probability=("abnormal_probability", "mean"),
            n_windows=("label", "size"),
        )
        .reset_index()
    )
    return grouped


def binary_metrics(
    y_true: Iterable[int], y_probability: Iterable[float], threshold: float = 0.5
) -> dict[str, Any]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_probability = np.asarray(list(y_probability), dtype=float)
    if y_true.shape != y_probability.shape:
        raise ValueError("Labels and probabilities must have identical shapes.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics for an empty set.")
    if not np.isfinite(y_probability).all():
        raise ValueError("Probabilities contain NaN or infinite values.")

    y_pred = (y_probability >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    metrics: dict[str, Any] = {
        "n_examples": len(y_true),
        "n_normal": int((y_true == 0).sum()),
        "n_abnormal": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_abnormal": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f2_abnormal": float(
            fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        # These must use continuous probabilities, never thresholded predictions.
        metrics["auroc"] = float(roc_auc_score(y_true, y_probability))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_probability))
    else:
        metrics["auroc"] = None
        metrics["pr_auc"] = None
    return metrics


def curve_values(
    y_true: Iterable[int], y_probability: Iterable[float]
) -> dict[str, np.ndarray]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_probability = np.asarray(list(y_probability), dtype=float)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probability)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_probability)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
    }
