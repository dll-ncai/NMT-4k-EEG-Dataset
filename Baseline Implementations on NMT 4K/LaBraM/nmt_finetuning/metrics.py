"""Window- and recording-level metrics for binary EEG abnormality detection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "threshold": float(threshold),
    }
    if np.unique(labels).size == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["pr_auc"] = float(average_precision_score(labels, probabilities))
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return result


def aggregate_recordings(
    labels: np.ndarray,
    probabilities: np.ndarray,
    recording_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grouped_probabilities: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for label, probability, recording_id in zip(labels, probabilities, recording_ids, strict=True):
        label_int = int(label)
        previous = grouped_labels.setdefault(recording_id, label_int)
        if previous != label_int:
            raise ValueError(f"Recording has conflicting labels: {recording_id}")
        grouped_probabilities[recording_id].append(float(probability))

    ordered_ids = sorted(grouped_probabilities)
    record_labels = np.asarray([grouped_labels[key] for key in ordered_ids], dtype=np.int64)
    record_probabilities = np.asarray(
        [np.mean(grouped_probabilities[key]) for key in ordered_ids], dtype=np.float64
    )
    return record_labels, record_probabilities, ordered_ids


def find_balanced_accuracy_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Tune a threshold on validation recordings only."""
    candidates = np.linspace(0.05, 0.95, 181)
    scores = np.asarray(
        [balanced_accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    )
    best_score = scores.max()
    best = candidates[np.flatnonzero(np.isclose(scores, best_score))]
    return float(best[np.argmin(np.abs(best - 0.5))])
