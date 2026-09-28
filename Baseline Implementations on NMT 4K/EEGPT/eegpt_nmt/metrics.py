"""Recording-level binary metrics and validation-only threshold selection."""

from __future__ import annotations

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
    """Compute all primary metrics using one prediction per recording."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError(f"Labels and probabilities have different shapes: {labels.shape}, {probabilities.shape}")
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, predictions, average="binary", zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if np.unique(labels).size == 2:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["auroc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    objective: str = "balanced_accuracy",
    minimum: float = 0.05,
    maximum: float = 0.95,
    steps: int = 181,
) -> tuple[float, dict[str, float]]:
    """Choose an operating threshold on validation recordings only."""
    allowed = {"balanced_accuracy", "macro_f1", "positive_f1"}
    if objective not in allowed:
        raise ValueError(f"Threshold objective must be one of {sorted(allowed)}, received {objective!r}.")
    candidates = np.linspace(minimum, maximum, steps)
    best_threshold = 0.5
    best_metrics = binary_metrics(labels, probabilities, best_threshold)
    best_key = (best_metrics[objective], -abs(best_threshold - 0.5))
    for threshold in candidates:
        current = binary_metrics(labels, probabilities, float(threshold))
        # The second key deterministically prefers a threshold closer to 0.5 on ties.
        current_key = (current[objective], -abs(float(threshold) - 0.5))
        if current_key > best_key:
            best_threshold, best_metrics, best_key = float(threshold), current, current_key
    return best_threshold, best_metrics


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    samples: int = 2000,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    """Estimate 95% recording-level confidence intervals by paired bootstrap."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    names = ["accuracy", "balanced_accuracy", "macro_f1", "positive_f1", "sensitivity", "specificity", "auroc", "pr_auc"]
    draws: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(int(samples)):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        result = binary_metrics(sampled_labels, probabilities[indices], threshold)
        for name in names:
            draws[name].append(result[name])
    intervals: dict[str, dict[str, float]] = {}
    for name, values in draws.items():
        if not values:
            intervals[name] = {"lower_95": float("nan"), "upper_95": float("nan")}
        else:
            lower, upper = np.percentile(values, [2.5, 97.5])
            intervals[name] = {"lower_95": float(lower), "upper_95": float(upper)}
    return intervals
