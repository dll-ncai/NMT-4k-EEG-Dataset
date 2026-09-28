"""Audit cohort completeness and verify that all three partitions are disjoint."""

from __future__ import annotations

import argparse

import pandas as pd

from .cohort import print_training_cohort_status, validate_manifest_for_training
from .config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--allow-incomplete-evaluation-for-training",
        action="store_true",
        help="Permit audit/training when the official training cohort is complete but evaluation EDFs are deferred.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.allow_incomplete_evaluation_for_training:
        config["training"]["allow_incomplete_evaluation_for_training"] = True
    manifest_path = resolve_path(config, config["paths"]["processed_manifest_csv"])
    assert manifest_path is not None
    frame = pd.read_csv(manifest_path)
    summary = validate_manifest_for_training(frame, config)
    print(frame.groupby(["experiment_split", "label"]).size())
    print()
    print_training_cohort_status(summary)
    print(f"All {len(frame)} currently processed recordings are unique and patient-disjoint.")


if __name__ == "__main__":
    main()
