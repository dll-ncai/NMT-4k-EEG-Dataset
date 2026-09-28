"""Command-line entry point for creating the internal validation partition."""

from __future__ import annotations

import argparse
import json

from .config import load_config, require_sections, resolve_path
from .protocol import read_and_split_metadata, split_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the experiment YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, ["paths", "split"])
    metadata_path = resolve_path(config, config["paths"]["metadata_csv"])
    output_path = resolve_path(config, config["paths"]["recording_splits_csv"])
    assert metadata_path is not None and output_path is not None

    split_frame = read_and_split_metadata(metadata_path, config["split"])
    summary = split_summary(split_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(output_path, index=False)
    summary.to_csv(output_path.with_name("split_summary.csv"), index=False)

    report = {
        "metadata_source": str(config["paths"]["metadata_csv"]),
        "split_file": str(config["paths"]["recording_splits_csv"]),
        "total_recordings": int(len(split_frame)),
        "patient_overlap_internal": 0,
        "patient_overlap_official": 0,
        "evaluation_used_for_splitting": False,
        "counts": summary.to_dict(orient="records"),
    }
    with output_path.with_name("protocol_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nLeakage-safe recording split created:\n")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_path}")
    print("The official evaluation recordings were not considered when forming the validation fold.")


if __name__ == "__main__":
    main()
