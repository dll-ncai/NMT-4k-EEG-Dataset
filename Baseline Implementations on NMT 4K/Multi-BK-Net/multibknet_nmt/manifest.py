from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import RuntimePaths

LABEL_TO_INDEX = {"normal": 0, "abnormal": 1}


def _edf_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        (
            path.resolve()
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".edf"
        ),
        key=lambda path: str(path).lower(),
    )


def discover_records(paths: RuntimePaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_dir in (
        ("train", paths.train_dir),
        ("evaluation", paths.evaluation_dir),
    ):
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing dataset partition directory: {split_dir}")
        for label, label_index in LABEL_TO_INDEX.items():
            class_dir = split_dir / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for edf_path in _edf_files(class_dir):
                rows.append(
                    {
                        "recording_id": edf_path.stem,
                        "split": split,
                        "label_name": label,
                        "label": label_index,
                        "edf_path": str(edf_path),
                        "relative_path": str(edf_path.relative_to(paths.dataset_root)),
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No EDF files were found beneath {paths.dataset_root}")

    duplicate_mask = frame["recording_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask, ["recording_id", "relative_path"]]
        detail = duplicates.to_string(index=False)
        raise RuntimeError(f"Duplicate recording identifiers were found:\n{detail}")

    return frame.sort_values(["split", "label", "recording_id"]).reset_index(drop=True)


def count_records(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "evaluation"):
        split_frame = frame[frame["split"] == split]
        result[split] = {
            "normal": int((split_frame["label"] == 0).sum()),
            "abnormal": int((split_frame["label"] == 1).sum()),
            "total": len(split_frame),
        }
    result["total"] = {
        "normal": int((frame["label"] == 0).sum()),
        "abnormal": int((frame["label"] == 1).sum()),
        "total": len(frame),
    }
    return result


def compare_expected_counts(
    actual: dict[str, dict[str, int]], expected: dict[str, dict[str, int]]
) -> list[str]:
    problems: list[str] = []
    for split in ("train", "evaluation"):
        for label in ("normal", "abnormal"):
            expected_value = int(expected[split][label])
            actual_value = int(actual[split][label])
            if expected_value != actual_value:
                problems.append(
                    f"{split}/{label}: expected {expected_value}, found {actual_value}"
                )
    return problems


def audit_metadata(metadata_path: Path, records: pd.DataFrame) -> dict[str, Any]:
    if not metadata_path.exists():
        return {
            "status": "missing",
            "path": str(metadata_path),
            "matched": 0,
            "missing_recordings": records["recording_id"].tolist(),
        }

    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    record_ids = set(records["recording_id"].astype(str))
    best_column: str | None = None
    best_ids: set[str] = set()
    best_overlap = -1

    for column in metadata.columns:
        values: set[str] = set()
        for value in metadata[column].dropna().astype(str):
            value = value.strip().replace("\\", "/")
            if not value:
                continue
            values.add(Path(value).stem)
        overlap = len(values & record_ids)
        if overlap > best_overlap:
            best_overlap = overlap
            best_column = str(column)
            best_ids = values

    missing = sorted(record_ids - best_ids)
    extras = sorted(best_ids - record_ids)
    status = "ok" if not missing else "recordings_missing_from_metadata"
    return {
        "status": status,
        "path": str(metadata_path),
        "rows": len(metadata),
        "identifier_column": best_column,
        "matched": int(best_overlap),
        "missing_recordings": missing,
        "metadata_ids_not_in_edf_tree": extras,
    }
