"""Cohort guards shared by audit and training entry points."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_manifest_for_training(
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, int | bool]:
    """Require complete training data while optionally deferring test EDFs.

    Missing official-evaluation recordings cannot influence optimization or
    internal validation: both datasets come exclusively from the official
    training partition. Final evaluation retains its exact-count guard.
    """
    required_columns = {
        "recording_id",
        "patient_id",
        "official_split",
        "experiment_split",
        "label",
        "tensor_path",
    }
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        raise RuntimeError(f"Processed manifest is missing required columns: {missing_columns}")
    if manifest["recording_id"].duplicated().any():
        examples = manifest.loc[
            manifest["recording_id"].duplicated(), "recording_id"
        ].head(10).tolist()
        raise RuntimeError(f"Duplicate recording IDs exist in the processed manifest: {examples}")

    recognized_official = manifest["official_split"].isin(["train", "evaluation"])
    if not recognized_official.all():
        unexpected = sorted(
            manifest.loc[~recognized_official, "official_split"].astype(str).unique()
        )
        raise RuntimeError(f"Unexpected official_split values in processed manifest: {unexpected}")

    expected_train = int(config["split"]["expected_train_recordings"])
    expected_evaluation = int(config["split"]["expected_evaluation_recordings"])
    actual_train = int((manifest["official_split"] == "train").sum())
    actual_evaluation = int((manifest["official_split"] == "evaluation").sum())
    allow_incomplete_evaluation = bool(
        config["training"].get("allow_incomplete_evaluation_for_training", False)
    )

    if actual_train != expected_train:
        raise RuntimeError(
            "Training is blocked because the official training cohort is incomplete: "
            f"expected {expected_train}, found {actual_train}."
        )
    if actual_evaluation > expected_evaluation:
        raise RuntimeError(
            f"Evaluation manifest contains {actual_evaluation} rows, exceeding "
            f"the expected {expected_evaluation}."
        )
    if actual_evaluation != expected_evaluation and not allow_incomplete_evaluation:
        raise RuntimeError(
            "Official evaluation is incomplete: "
            f"expected {expected_evaluation}, found {actual_evaluation}. "
            "Restore the missing EDFs or explicitly set "
            "training.allow_incomplete_evaluation_for_training=true."
        )

    experiment_training_rows = int(
        manifest["experiment_split"].isin(["internal_train", "validation"]).sum()
    )
    misplaced_training_rows = manifest[
        (manifest["official_split"] == "train")
        & ~manifest["experiment_split"].isin(["internal_train", "validation"])
    ]
    if experiment_training_rows != expected_train or not misplaced_training_rows.empty:
        raise RuntimeError(
            "The complete official training cohort was not assigned exclusively "
            "to internal_train/validation."
        )
    misplaced_evaluation_rows = manifest[
        (manifest["official_split"] == "evaluation")
        & (manifest["experiment_split"] != "evaluation")
    ]
    if not misplaced_evaluation_rows.empty:
        raise RuntimeError("An official evaluation row was assigned to training or validation.")

    patient_sets = {
        split: set(
            manifest.loc[manifest["experiment_split"] == split, "patient_id"].astype(str)
        )
        for split in ("internal_train", "validation", "evaluation")
    }
    for left, right in (
        ("internal_train", "validation"),
        ("internal_train", "evaluation"),
        ("validation", "evaluation"),
    ):
        overlap = sorted(patient_sets[left].intersection(patient_sets[right]))
        if overlap:
            raise RuntimeError(f"Patient overlap between {left} and {right}: {overlap[:10]}")

    return {
        "expected_train": expected_train,
        "actual_train": actual_train,
        "expected_evaluation": expected_evaluation,
        "actual_evaluation": actual_evaluation,
        "missing_evaluation": expected_evaluation - actual_evaluation,
        "allow_incomplete_evaluation": allow_incomplete_evaluation,
    }


def print_training_cohort_status(summary: dict[str, int | bool]) -> None:
    """Print an unmistakable distinction between training and test readiness."""
    print(
        "Official training cohort: "
        f"{summary['actual_train']}/{summary['expected_train']} (training-ready)"
    )
    print(
        "Official evaluation cohort: "
        f"{summary['actual_evaluation']}/{summary['expected_evaluation']}"
    )
    if int(summary["missing_evaluation"]) > 0:
        print(
            "WARNING: Training/internal validation may proceed, but final evaluation "
            f"remains blocked until the {summary['missing_evaluation']} missing "
            "evaluation recording(s) are restored."
        )
    else:
        print("Final-evaluation cohort is complete.")
