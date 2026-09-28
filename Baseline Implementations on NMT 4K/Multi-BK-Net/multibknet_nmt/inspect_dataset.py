from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import pandas as pd
from tqdm import tqdm

from .channels import canonicalize_channel, select_channel_indices
from .config import ensure_output_directories, load_config, resolved_config
from .manifest import (
    audit_metadata,
    compare_expected_counts,
    count_records,
    discover_records,
)
from .utils import atomic_write_json, print_heading


def inspect(config_path: str | Path, limit: int | None = None) -> int:
    config, paths = load_config(config_path)
    ensure_output_directories(paths)
    required_channels = list(config["dataset"]["channels"])

    print_heading("NMT-4K dataset discovery")
    records = discover_records(paths)
    actual_counts = count_records(records)
    count_problems = compare_expected_counts(
        actual_counts, config["dataset"]["expected_counts"]
    )
    print(json.dumps(actual_counts, indent=2))
    for problem in count_problems:
        print(f"COUNT ERROR: {problem}")

    metadata_audit = audit_metadata(paths.metadata_tsv, records)
    if metadata_audit["status"] != "ok":
        print(f"METADATA ERROR: {metadata_audit['status']}")

    scan_records = records if limit is None else records.head(limit)
    inventory_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    channel_counter: Counter[str] = Counter()

    print_heading("EDF header inspection")
    mne.set_log_level("ERROR")
    for row in tqdm(
        scan_records.itertuples(index=False), total=len(scan_records), unit="recording"
    ):
        result = row._asdict()
        try:
            raw = mne.io.read_raw_edf(row.edf_path, preload=False, verbose="ERROR")
            for name in raw.ch_names:
                channel_counter[canonicalize_channel(name) or f"UNMAPPED:{name}"] += 1
            _, mapping = select_channel_indices(raw.ch_names, required_channels)
            sfreq = float(raw.info["sfreq"])
            duration_seconds = float(raw.n_times / sfreq)
            result.update(
                {
                    "status": "ok",
                    "sfreq": sfreq,
                    "duration_seconds": duration_seconds,
                    "raw_channel_count": len(raw.ch_names),
                    "selected_channel_count": len(mapping),
                    "selected_channel_mapping": json.dumps(mapping, sort_keys=True),
                    "error": "",
                }
            )
            raw.close()
        except Exception as exc:  # noqa: BLE001 - every EDF failure is audited independently
            result.update(
                {
                    "status": "failed",
                    "sfreq": None,
                    "duration_seconds": None,
                    "raw_channel_count": None,
                    "selected_channel_count": None,
                    "selected_channel_mapping": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failures.append(result.copy())
        inventory_rows.append(result)

    inventory = pd.DataFrame(inventory_rows)
    inventory_path = paths.audit_dir / "dataset_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    pd.DataFrame(failures).to_csv(
        paths.audit_dir / "inspection_failures.csv", index=False
    )
    pd.DataFrame(
        [
            {"channel": channel, "recording_count": count}
            for channel, count in channel_counter.most_common()
        ]
    ).to_csv(paths.audit_dir / "channel_counts.csv", index=False)

    durations = inventory.loc[inventory["status"] == "ok", "duration_seconds"].dropna()
    sfreqs = inventory.loc[inventory["status"] == "ok", "sfreq"].dropna()
    summary = {
        "status": "ok",
        "scan_limit": limit,
        "counts": actual_counts,
        "count_errors": count_problems,
        "metadata": metadata_audit,
        "recordings_scanned": len(inventory),
        "recordings_failed": len(failures),
        "sampling_rates": sorted(float(value) for value in sfreqs.unique()),
        "duration_seconds": {
            "minimum": float(durations.min()) if len(durations) else None,
            "median": float(durations.median()) if len(durations) else None,
            "maximum": float(durations.max()) if len(durations) else None,
        },
        "required_channels": required_channels,
        "resolved_config": resolved_config(config, paths),
    }

    blocking = bool(count_problems or failures or metadata_audit["status"] != "ok")
    if limit is not None:
        # A quick scan cannot certify the complete dataset.
        blocking = False
        summary["status"] = "quick_scan_only"
    elif blocking:
        summary["status"] = "blocking_errors"
    atomic_write_json(paths.audit_dir / "dataset_summary.json", summary)

    print(f"Inventory: {inventory_path}")
    print(f"Failed EDF headers: {len(failures)}")
    print(f"Final status: {summary['status']}")
    return 1 if blocking else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect NMT-4K EDF structure and headers."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--limit", type=int, default=None, help="Quick non-certifying header scan."
    )
    args = parser.parse_args()
    raise SystemExit(inspect(args.config, args.limit))


if __name__ == "__main__":
    main()
