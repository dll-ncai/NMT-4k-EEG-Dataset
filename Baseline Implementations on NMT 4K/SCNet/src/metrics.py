from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def binary_metrics(y_true: np.ndarray, prob_abnormal: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_abnormal = np.asarray(prob_abnormal, dtype=np.float64)
    y_pred = (prob_abnormal >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auroc = roc_auc_score(y_true, prob_abnormal)
    except ValueError:
        auroc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": float(auroc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true: np.ndarray, prob_abnormal: np.ndarray, objective: str = "f1") -> tuple[float, dict]:
    """Tune threshold only on validation predictions.

    Searches a dense deterministic grid. ``objective`` may be ``f1`` or ``youden``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_abnormal = np.asarray(prob_abnormal, dtype=np.float64)
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -np.inf
    best_metrics = None
    for t in thresholds:
        m = binary_metrics(y_true, prob_abnormal, float(t))
        if objective == "f1":
            score = m["f1"]
        elif objective == "youden":
            score = m["sensitivity"] + m["specificity"] - 1.0
        else:
            raise ValueError("threshold objective must be 'f1' or 'youden'")
        # Deterministic tie break: prefer the threshold closest to 0.5.
        if score > best_score or (
            np.isclose(score, best_score) and abs(t - 0.5) < abs(best_t - 0.5)
        ):
            best_score = score
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics
