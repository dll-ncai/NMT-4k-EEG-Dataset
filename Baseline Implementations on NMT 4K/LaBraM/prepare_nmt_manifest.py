#!/usr/bin/env python
"""Fast manifest builder using pre-generated inspection report CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path(r"E:\LaBraM-NMT\data\windows_inspection_report.csv"),
        help="Path to the source inspection report CSV",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"G:\NMT_processed_4500"),
        help="Root directory where .pkl files are physically located",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output manifest CSV",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def normalize_rel_path(raw_path: str) -> str:
    """Strip leading folder roots to align with target data-dir relative structure."""
    p = Path(raw_path)
    parts = p.parts
    # Strip prefixes like 'sed_4500' if present in the CSV relative path
    if parts and parts[0].lower() in {"sed_4500", "nmt_processed_4500"}:
        parts = parts[1:]
    return str(Path(*parts))


def main() -> None:
    args = parse_args()
    report_csv = args.report_csv.resolve()
    data_dir = args.data_dir.resolve()
    output_path = args.output or (data_dir / "nmt_finetune_manifest.csv")

    if not report_csv.exists():
        raise FileNotFoundError(f"Inspection report not found at: {report_csv}")

    print(f"Reading labels directly from: {report_csv}")

    raw_rows = []
    labels_by_recording: dict[str, int] = {}
    source_split_by_recording: dict[str, str] = {}

    with report_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip invalid window records if flagged in report
            if row.get("status", "").lower() == "invalid" or row.get("error", "").lower() == "invalid":
                continue

            rel_path = normalize_rel_path(row["file_path"] if "file_path" in row else row.get("path", ""))
            full_path = str(data_dir / rel_path)

            rec_id = row["recording_id"]
            lbl = int(row["label"])
            
            # Map eval/evaluation -> test, otherwise train
            raw_split = row.get("source_split", row.get("split", "")).lower()
            source_split = "test" if raw_split in {"eval", "evaluation", "test"} else "train"

            labels_by_recording[rec_id] = lbl
            source_split_by_recording.setdefault(rec_id, source_split)

            raw_rows.append({
                "file_path": full_path,
                "source_split": source_split,
                "label": lbl,
                "recording_id": rec_id,
                "window_index": int(row["window_index"]),
            })

    print(f"Loaded {len(raw_rows)} valid window records across {len(labels_by_recording)} unique recordings.")

    # Group train recordings for stratified validation splitting
    train_recs = [r for r, s in source_split_by_recording.items() if s == "train"]
    grouped_train: dict[int, list[str]] = defaultdict(list)
    for r in train_recs:
        grouped_train[labels_by_recording[r]].append(r)

    rng = random.Random(args.seed)
    val_recs: set[str] = set()

    for lbl_val, rec_list in grouped_train.items():
        s_ids = sorted(rec_list)
        rng.shuffle(s_ids)
        val_count = max(1, round(len(s_ids) * args.val_fraction))
        val_recs.update(s_ids[:val_count])

    for row in raw_rows:
        if row["source_split"] == "test":
            row["split"] = "test"
        elif row["recording_id"] in val_recs:
            row["split"] = "val"
        else:
            row["split"] = "train"

    raw_rows.sort(key=lambda x: (x["split"], x["recording_id"], x["window_index"]))

    print(f"Writing manifest CSV to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path", "split", "label", "recording_id", "window_index"]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in raw_rows if r["split"] == split]
        recs = {(r["recording_id"], r["label"]) for r in subset}
        summary[split] = {
            "windows": len(subset),
            "recordings": len(recs),
            "normal_recordings": sum(1 for _, l in recs if l == 0),
            "abnormal_recordings": sum(1 for _, l in recs if l == 1),
        }

    print("\nFinal Dataset Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()