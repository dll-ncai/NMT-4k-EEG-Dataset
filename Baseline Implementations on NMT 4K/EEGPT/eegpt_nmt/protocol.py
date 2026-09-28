"""Metadata normalization and leakage-safe train/validation splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column that exists, ignoring capitalization."""
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _parse_binary_label(value: Any) -> int:
    """Convert numeric or textual Normal/Abnormal labels to 0/1."""
    if pd.isna(value):
        raise ValueError("A metadata row has a missing label.")
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, (float, np.floating)) and float(value).is_integer()
    ):
        parsed = int(value)
    else:
        normalized = str(value).strip().lower()
        mapping = {"normal": 0, "n": 0, "0": 0, "abnormal": 1, "a": 1, "1": 1}
        if normalized not in mapping:
            raise ValueError(f"Unrecognized label: {value!r}")
        parsed = mapping[normalized]
    if parsed not in (0, 1):
        raise ValueError(f"Binary labels must be 0 or 1, received {parsed}.")
    return parsed


def _parse_official_split(value: Any) -> str:
    """Map harmless spelling variants to the two official NMT partitions."""
    normalized = str(value).strip().lower()
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"evaluation", "eval", "test", "testing"}:
        return "evaluation"
    raise ValueError(f"Unrecognized official split: {value!r}")


def canonicalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one canonical row per recording from common NMT column names."""
    recording_col = _first_existing_column(frame, ["recording_id", "File Name", "recordname"])
    label_col = _first_existing_column(frame, ["label", "Label (Normal/Abnormal)", "label_new"])
    split_col = _first_existing_column(frame, ["official_split", "split"])
    path_col = _first_existing_column(frame, ["file_path", "path", "edf_path"])
    if recording_col is None or label_col is None or split_col is None:
        raise KeyError("Metadata must contain recording ID, label, and official split columns.")

    canonical = pd.DataFrame()
    canonical["recording_id"] = frame[recording_col].astype(str).str.strip()
    patient_col = _first_existing_column(frame, ["patient_id", "subject_id"])
    if patient_col:
        patient_values = frame[patient_col]
        valid_patient = patient_values.notna() & patient_values.astype(str).str.strip().ne("")
        canonical["patient_id"] = patient_values.astype(str).str.strip().where(
            valid_patient, canonical["recording_id"]
        )
    else:
        canonical["patient_id"] = canonical["recording_id"]
    canonical["label"] = frame[label_col].map(_parse_binary_label).astype(int)
    canonical["official_split"] = frame[split_col].map(_parse_official_split)
    canonical["source_file_path"] = frame[path_col].astype(str) if path_col else ""

    optional_columns = {
        "hospital": ["Hospital", "hospital", "site"],
        "gender": ["gender", "Gender", "sex"],
        "age_years": ["Age (Years)", "age_years", "age"],
        "year": ["year", "Year"],
    }
    for output_name, candidates in optional_columns.items():
        source = _first_existing_column(frame, candidates)
        canonical[output_name] = frame[source] if source else "unknown"

    if canonical["recording_id"].eq("").any():
        raise ValueError("At least one recording ID is empty.")
    duplicates = canonical[canonical["recording_id"].duplicated(keep=False)]
    if not duplicates.empty:
        examples = duplicates["recording_id"].unique()[:10].tolist()
        raise ValueError(f"Recording IDs are not unique. Examples: {examples}")
    return canonical


def validate_official_partition(
    metadata: pd.DataFrame,
    expected_train: int | None,
    expected_evaluation: int | None,
) -> None:
    """Detect cohort changes and subject leakage before creating a validation set."""
    counts = metadata["official_split"].value_counts().to_dict()
    if expected_train is not None and counts.get("train", 0) != int(expected_train):
        raise ValueError(
            f"Expected {expected_train} official training recordings, found {counts.get('train', 0)}."
        )
    if expected_evaluation is not None and counts.get("evaluation", 0) != int(expected_evaluation):
        raise ValueError(
            f"Expected {expected_evaluation} evaluation recordings, found {counts.get('evaluation', 0)}."
        )

    train_patients = set(metadata.loc[metadata["official_split"] == "train", "patient_id"])
    eval_patients = set(metadata.loc[metadata["official_split"] == "evaluation", "patient_id"])
    overlap = sorted(train_patients.intersection(eval_patients))
    if overlap:
        raise ValueError(f"Patient leakage across official train/evaluation partitions: {overlap[:10]}")


def create_experiment_splits(
    metadata: pd.DataFrame,
    n_folds: int,
    validation_fold: int,
    seed: int,
    stratify_by_hospital: bool = True,
) -> pd.DataFrame:
    """Split only official training subjects; preserve evaluation untouched."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if not 0 <= validation_fold < n_folds:
        raise ValueError(f"validation_fold must be between 0 and {n_folds - 1}.")

    output = metadata.copy()
    output["experiment_split"] = "evaluation"
    official_train = output[output["official_split"] == "train"].copy()

    label_text = official_train["label"].astype(str)
    if stratify_by_hospital and "hospital" in official_train:
        composite = label_text + "__" + official_train["hospital"].fillna("unknown").astype(str)
        # Each stratum needs enough patients to distribute across all folds.
        if composite.value_counts().min() >= n_folds:
            stratification_target = composite
        else:
            stratification_target = label_text
    else:
        stratification_target = label_text

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    chosen_train_indices: np.ndarray | None = None
    chosen_val_indices: np.ndarray | None = None
    dummy_x = np.zeros(len(official_train), dtype=np.uint8)
    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(dummy_x, stratification_target, groups=official_train["patient_id"])
    ):
        if fold == validation_fold:
            chosen_train_indices, chosen_val_indices = train_indices, val_indices
            break

    assert chosen_train_indices is not None and chosen_val_indices is not None
    train_ids = set(official_train.iloc[chosen_train_indices]["recording_id"])
    validation_ids = set(official_train.iloc[chosen_val_indices]["recording_id"])
    output.loc[output["recording_id"].isin(train_ids), "experiment_split"] = "internal_train"
    output.loc[output["recording_id"].isin(validation_ids), "experiment_split"] = "validation"

    train_patients = set(output.loc[output["experiment_split"] == "internal_train", "patient_id"])
    val_patients = set(output.loc[output["experiment_split"] == "validation", "patient_id"])
    if train_patients.intersection(val_patients):
        raise AssertionError("Internal patient split failed: train and validation overlap.")
    if not output.loc[output["official_split"] == "evaluation", "experiment_split"].eq("evaluation").all():
        raise AssertionError("The official evaluation partition was modified.")
    return output


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an easy-to-audit count table by experimental split and label."""
    return (
        frame.groupby(["experiment_split", "label"], dropna=False)
        .size()
        .rename("recordings")
        .reset_index()
        .sort_values(["experiment_split", "label"])
    )


def read_and_split_metadata(metadata_path: Path, split_options: dict[str, Any]) -> pd.DataFrame:
    """Convenience entry point used by the command-line script and tests."""
    raw = pd.read_csv(metadata_path)
    canonical = canonicalize_metadata(raw)
    validate_official_partition(
        canonical,
        expected_train=split_options.get("expected_train_recordings"),
        expected_evaluation=split_options.get("expected_evaluation_recordings"),
    )
    return create_experiment_splits(
        canonical,
        n_folds=int(split_options.get("n_folds", 5)),
        validation_fold=int(split_options.get("validation_fold", 0)),
        seed=int(split_options.get("seed", 2026)),
        stratify_by_hospital=bool(split_options.get("stratify_by_hospital", True)),
    )
