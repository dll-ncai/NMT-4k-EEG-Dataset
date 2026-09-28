"""Protected recording-level evaluation of the validation-selected checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from .checkpoint import load_experiment_checkpoint
from .config import load_config, require_sections, resolve_path
from .data import RecordingBagDataset
from .metrics import binary_metrics, bootstrap_confidence_intervals
from .model import EEGPTMILClassifier
from .train import predict_recordings, set_reproducibility


def _save_plots(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output: Path,
    *,
    provisional: bool,
) -> None:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Abnormal"])
    display.plot(cmap="Blues", colorbar=False)
    status = "Provisional " if provisional else ""
    plt.title(f"{status}recording-level confusion matrix (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix_recording_level.png", dpi=300)
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False-positive rate")
    plt.ylabel("Sensitivity")
    plt.title(f"{status}recording-level ROC curve")
    plt.tight_layout()
    plt.savefig(output / "roc_curve_recording_level.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall / sensitivity")
    plt.ylabel("Precision")
    plt.title(f"{status}recording-level precision-recall curve")
    plt.tight_layout()
    plt.savefig(output / "pr_curve_recording_level.png", dpi=300)
    plt.close()


def _validate_evaluation_cohort(
    actual_count: int,
    expected_official_count: int,
    *,
    provisional: bool,
    expected_available_records: int | None,
) -> str:
    """Validate official or explicitly acknowledged incomplete evaluation."""
    if actual_count > expected_official_count:
        raise RuntimeError(
            "Evaluation manifest exceeds the official cohort: "
            f"expected at most {expected_official_count}, found {actual_count}."
        )

    if provisional:
        if expected_available_records is None:
            raise RuntimeError(
                "Provisional evaluation requires --expected-available-records so the "
                "incomplete cohort size is acknowledged explicitly."
            )
        if expected_available_records != actual_count:
            raise RuntimeError(
                "Provisional cohort count did not match the explicit acknowledgment: "
                f"expected {expected_available_records}, found {actual_count}."
            )
        if actual_count == expected_official_count:
            raise RuntimeError(
                "The complete official evaluation cohort is available. Use "
                "--confirm-final-test instead of provisional mode."
            )
        return "provisional_incomplete_cohort"

    if expected_available_records is not None:
        raise RuntimeError(
            "--expected-available-records is valid only with "
            "--confirm-provisional-incomplete-test."
        )
    if actual_count != expected_official_count:
        raise RuntimeError(
            f"Final evaluation requires exactly {expected_official_count} recordings, "
            f"found {actual_count}. Use the explicit provisional mode only if the "
            "incomplete result will not be reported as the official benchmark."
        )
    return "official_complete_cohort"


def _find_missing_evaluation_records(
    config: dict[str, Any],
    present_recording_ids: set[str],
) -> list[dict[str, Any]]:
    """Return missing official-evaluation IDs from the frozen split artifact."""
    split_path = resolve_path(config, config["paths"].get("recording_splits_csv"))
    if split_path is None or not split_path.exists():
        return []
    split_frame = pd.read_csv(split_path)
    required = {"recording_id", "official_split"}
    if not required.issubset(split_frame.columns):
        return []
    official_evaluation = split_frame.loc[
        split_frame["official_split"].astype(str) == "evaluation"
    ].copy()
    official_evaluation["recording_id"] = official_evaluation["recording_id"].astype(str)
    missing = official_evaluation.loc[
        ~official_evaluation["recording_id"].isin(present_recording_ids)
    ]
    columns = [column for column in ("recording_id", "label") if column in missing.columns]
    records: list[dict[str, Any]] = []
    for row in missing[columns].to_dict(orient="records"):
        clean_row: dict[str, Any] = {"recording_id": str(row["recording_id"])}
        if "label" in row and not pd.isna(row["label"]):
            clean_row["label"] = int(row["label"])
            clean_row["class_name"] = "abnormal" if int(row["label"]) == 1 else "normal"
        records.append(clean_row)
    return records


def evaluate(
    config: dict[str, Any],
    *,
    provisional: bool = False,
    expected_available_records: int | None = None,
) -> dict[str, Any]:
    paths = config["paths"]
    options = config["evaluation"]
    seed = int(config["training"].get("seed", 2026))
    set_reproducibility(seed, bool(config["training"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"

    manifest_path = resolve_path(config, paths["processed_manifest_csv"])
    runs_directory = resolve_path(config, paths["runs_dir"])
    output_root = resolve_path(config, paths["evaluation_output_dir"])
    assert manifest_path is not None and runs_directory is not None and output_root is not None
    run_name = str(config["training"].get("run_name", f"eegpt_nmt_v2_seed{seed}"))
    checkpoint_path = runs_directory / run_name / "best.pt"

    evaluation_dataset = RecordingBagDataset(
        manifest_path,
        config["_project_root"],
        split="evaluation",
        windows_per_recording=int(options.get("windows_per_recording", 0)),
        training=False,
        seed=seed,
    )
    expected_count = int(config["split"].get("expected_evaluation_recordings", 1000))
    actual_count = int(len(evaluation_dataset))
    evaluation_status = _validate_evaluation_cohort(
        actual_count,
        expected_count,
        provisional=provisional,
        expected_available_records=expected_available_records,
    )
    missing_records = _find_missing_evaluation_records(
        config,
        set(evaluation_dataset.records["recording_id"].astype(str)),
    )
    available_labels = evaluation_dataset.records["label"].astype(int)
    available_normal = int((available_labels == 0).sum())
    available_abnormal = int((available_labels == 1).sum())
    if provisional:
        output_directory = (
            output_root
            / "provisional_incomplete_cohort"
            / f"{run_name}_{actual_count}of{expected_count}"
        )
        print("\n" + "!" * 78)
        print("PROVISIONAL INCOMPLETE-COHORT EVALUATION")
        print(f"Available evaluation recordings: {actual_count}/{expected_count}")
        print(
            "Available class counts: "
            f"normal={available_normal}, abnormal={available_abnormal}"
        )
        print(f"Missing evaluation recordings: {expected_count - actual_count}")
        if missing_records:
            missing_text = ", ".join(
                f"{row['recording_id']} ({row.get('class_name', 'class unknown')})"
                for row in missing_records
            )
            print(f"Missing IDs: {missing_text}")
        print("This output is not the official 1,000-record NMT-4K benchmark.")
        print("The validation-selected checkpoint and threshold must remain frozen.")
        print("!" * 78 + "\n")
    else:
        output_directory = output_root / run_name
    output_directory.mkdir(parents=True, exist_ok=True)

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = EEGPTMILClassifier(**raw_checkpoint["model_options"])
    checkpoint = load_experiment_checkpoint(model, checkpoint_path)
    model.to(device)

    if int(options.get("windows_per_recording", 0)) == 0:
        evaluation_batch_size = 1
    else:
        evaluation_batch_size = int(options.get("recording_batch_size", 1))
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=int(options.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    predictions = predict_recordings(
        model,
        evaluation_loader,
        device,
        amp_enabled,
        encode_chunk_size=int(options.get("encode_chunk_size", 16)),
    )
    metadata_columns = [column for column in ("recording_id", "hospital", "gender", "age_years", "year") if column in evaluation_dataset.records]
    predictions = predictions.merge(evaluation_dataset.records[metadata_columns], on="recording_id", how="left", validate="one_to_one")
    labels = predictions["label"].to_numpy(dtype=int)
    probabilities = predictions["probability_abnormal"].to_numpy(dtype=float)
    validation_threshold = float(checkpoint["validation_threshold"])
    metrics_at_validation_threshold = binary_metrics(labels, probabilities, validation_threshold)
    metrics_at_half = binary_metrics(labels, probabilities, 0.5)
    intervals = bootstrap_confidence_intervals(
        labels,
        probabilities,
        validation_threshold,
        samples=int(options.get("bootstrap_samples", 2000)),
        seed=seed,
    )

    predictions["prediction_at_validation_threshold"] = (probabilities >= validation_threshold).astype(int)
    predictions_filename = (
        "provisional_recording_predictions.csv" if provisional else "recording_predictions.csv"
    )
    predictions.to_csv(output_directory / predictions_filename, index=False)
    _save_plots(
        labels,
        probabilities,
        validation_threshold,
        output_directory,
        provisional=provisional,
    )
    normal_count = int((labels == 0).sum())
    abnormal_count = int((labels == 1).sum())
    report = {
        "evaluation_status": evaluation_status,
        "official_benchmark_complete": not provisional,
        "evaluation_unit": "recording",
        "number_of_recordings": actual_count,
        "expected_official_recordings": expected_count,
        "missing_recording_count": expected_count - actual_count,
        "available_class_counts": {
            "normal": normal_count,
            "abnormal": abnormal_count,
        },
        "missing_evaluation_records": missing_records,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold_source": "internal validation partition",
        "validation_selected_threshold": validation_threshold,
        "metrics_at_validation_threshold": metrics_at_validation_threshold,
        "metrics_at_0.5": metrics_at_half,
        "bootstrap_95_percent_intervals": intervals,
    }
    metrics_filename = "provisional_metrics.json" if provisional else "final_metrics.json"
    with (output_directory / metrics_filename).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    heading = (
        f"PROVISIONAL {actual_count}-RECORD NMT-4K RESULTS"
        if provisional
        else "FINAL RECORDING-LEVEL NMT-4K RESULTS"
    )
    print(f"\n{heading}")
    print(f"Recordings: {len(predictions)}; threshold selected on validation: {validation_threshold:.3f}")
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "positive_f1", "sensitivity", "specificity", "auroc", "pr_auc"):
        print(f"{name:>20}: {metrics_at_validation_threshold[name]:.4f}")
    print(f"\nSaved results: {output_directory}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    confirmation = parser.add_mutually_exclusive_group(required=True)
    confirmation.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required acknowledgment that hyperparameters and threshold are already fixed.",
    )
    confirmation.add_argument(
        "--confirm-provisional-incomplete-test",
        action="store_true",
        help=(
            "Explicitly permit a clearly labeled incomplete-cohort evaluation; "
            "does not produce an official benchmark result."
        ),
    )
    parser.add_argument(
        "--expected-available-records",
        type=int,
        default=None,
        help="Exact available cohort size; required only for provisional evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confirm_provisional_incomplete_test and args.expected_available_records is None:
        raise SystemExit(
            "Provisional evaluation requires --expected-available-records with the exact "
            "processed evaluation count."
        )
    if args.confirm_final_test and args.expected_available_records is not None:
        raise SystemExit(
            "--expected-available-records may be used only with "
            "--confirm-provisional-incomplete-test."
        )
    config = load_config(args.config)
    require_sections(config, ["paths", "split", "model", "training", "evaluation"])
    evaluate(
        config,
        provisional=bool(args.confirm_provisional_incomplete_test),
        expected_available_records=args.expected_available_records,
    )


if __name__ == "__main__":
    main()
