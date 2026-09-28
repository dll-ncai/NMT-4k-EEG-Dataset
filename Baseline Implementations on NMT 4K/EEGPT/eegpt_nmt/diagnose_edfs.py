"""Inspect failed NMT EDF containers without modifying the source recordings.

The report separates optional-header parser problems from incomplete signal
payloads. It intentionally never repairs an EDF in place: a missing waveform
must be replaced from a verified dataset copy, not synthesized.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import mne
import pandas as pd

from .config import load_config, require_sections, resolve_path


def _ascii(value: bytes) -> str:
    return value.decode("latin-1", errors="replace").split("\x00")[0].strip()


def _parse_number(value: bytes, converter: Callable[[str], Any]) -> Any | None:
    text = _ascii(value).replace(",", ".")
    if not text:
        return None
    try:
        return converter(text)
    except ValueError:
        return None


def _signal_fields(header: bytes, n_signals: int) -> dict[str, list[bytes]]:
    """Return the EDF per-signal header fields using the standard byte layout."""
    offset = 256
    fields: dict[str, list[bytes]] = {}

    def take(name: str, width: int) -> None:
        nonlocal offset
        fields[name] = [
            header[offset + index * width : offset + (index + 1) * width]
            for index in range(n_signals)
        ]
        offset += width * n_signals

    take("label", 16)
    take("transducer", 80)
    take("physical_dimension", 8)
    take("physical_min", 8)
    take("physical_max", 8)
    take("digital_min", 8)
    take("digital_max", 8)
    take("prefilter", 80)
    take("samples_per_record", 8)
    take("reserved", 32)
    return fields


def inspect_edf(edf_path: Path) -> dict[str, Any]:
    """Return structural and MNE-reader diagnostics for one EDF file."""
    result: dict[str, Any] = {
        "edf_path": str(edf_path),
        "file_size_bytes": edf_path.stat().st_size,
        "mne_version": mne.__version__,
    }
    with edf_path.open("rb") as stream:
        fixed_header = stream.read(256)

    if len(fixed_header) < 256:
        result.update(
            status="truncated_fixed_header",
            structural_issues=f"File has only {len(fixed_header)} of the required 256 fixed-header bytes.",
        )
        return result

    header_nbytes = _parse_number(fixed_header[184:192], int)
    declared_records = _parse_number(fixed_header[236:244], int)
    record_duration_seconds = _parse_number(fixed_header[244:252], float)
    n_signals = _parse_number(fixed_header[252:256], int)
    patient_tokens = _ascii(fixed_header[8:88]).split()
    invalid_optional_patient_fields: list[str] = []
    for token in patient_tokens[4:]:
        if "=" not in token:
            continue
        key, value = token.split("=", maxsplit=1)
        converter: Callable[[str], Any] | None = float if key in {"weight", "height"} else (
            int if key == "hand" else None
        )
        if converter is not None:
            try:
                converter(value)
            except ValueError:
                invalid_optional_patient_fields.append(f"{key}={value!r}")

    result.update(
        header_nbytes=header_nbytes,
        declared_data_records=declared_records,
        record_duration_seconds=record_duration_seconds,
        n_signals=n_signals,
        invalid_optional_patient_fields="; ".join(invalid_optional_patient_fields),
    )
    structural_issues: list[str] = []
    if header_nbytes is None or header_nbytes < 256:
        structural_issues.append("invalid fixed-header byte count")
    if declared_records is None:
        structural_issues.append("blank/invalid number of data records")
    if record_duration_seconds is None or record_duration_seconds <= 0:
        structural_issues.append("blank/invalid data-record duration")
    if n_signals is None or n_signals <= 0:
        structural_issues.append("blank/invalid number of signals")

    if header_nbytes is not None and header_nbytes > result["file_size_bytes"]:
        structural_issues.append(
            f"declared header ({header_nbytes} bytes) exceeds file size ({result['file_size_bytes']} bytes)"
        )

    if (
        header_nbytes is not None
        and n_signals is not None
        and n_signals > 0
        and result["file_size_bytes"] >= header_nbytes
    ):
        with edf_path.open("rb") as stream:
            header = stream.read(header_nbytes)
        expected_header_nbytes = 256 + 256 * n_signals
        if header_nbytes != expected_header_nbytes:
            structural_issues.append(
                f"header size is {header_nbytes}; standard size for {n_signals} signals is {expected_header_nbytes}"
            )
        fields = _signal_fields(header, n_signals)
        invalid_calibration_fields: list[str] = []
        for field_name in ("physical_min", "physical_max", "digital_min", "digital_max"):
            for index, value in enumerate(fields[field_name]):
                if _parse_number(value, float) is None:
                    invalid_calibration_fields.append(f"{field_name}[{index}]")
        samples_per_record = [
            _parse_number(value, int) for value in fields["samples_per_record"]
        ]
        invalid_sample_fields = [
            index for index, value in enumerate(samples_per_record) if value is None or value <= 0
        ]
        result["invalid_calibration_fields"] = "; ".join(invalid_calibration_fields)
        result["invalid_samples_per_record_indices"] = "; ".join(map(str, invalid_sample_fields))
        if invalid_calibration_fields:
            structural_issues.append(
                "blank/invalid signal calibration fields: " + ", ".join(invalid_calibration_fields)
            )
        if invalid_sample_fields:
            structural_issues.append(
                "blank/invalid samples-per-record fields at signal indices: "
                + ", ".join(map(str, invalid_sample_fields))
            )

        if not invalid_sample_fields:
            bytes_per_data_record = 2 * sum(int(value) for value in samples_per_record if value is not None)
            data_bytes = result["file_size_bytes"] - header_nbytes
            complete_records = data_bytes // bytes_per_data_record if bytes_per_data_record else 0
            trailing_bytes = data_bytes % bytes_per_data_record if bytes_per_data_record else data_bytes
            result.update(
                bytes_per_data_record=bytes_per_data_record,
                data_bytes=data_bytes,
                complete_data_records=complete_records,
                trailing_data_bytes=trailing_bytes,
            )
            if data_bytes <= 0:
                structural_issues.append("EDF has a header but no waveform data bytes")
            elif complete_records == 0:
                structural_issues.append("EDF does not contain one complete waveform data record")
            if declared_records is not None and declared_records >= 0 and complete_records < declared_records:
                structural_issues.append(
                    f"file contains {complete_records} complete records but header declares {declared_records}"
                )
            if trailing_bytes:
                structural_issues.append(f"waveform payload ends with {trailing_bytes} incomplete bytes")

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        result.update(
            mne_header_status="readable",
            mne_n_times=int(raw.n_times),
            mne_sfreq=float(raw.info["sfreq"]),
            mne_n_channels=len(raw.ch_names),
        )
        if raw.n_times <= 0:
            structural_issues.append("MNE found zero readable samples")
    except Exception as error:
        result.update(
            mne_header_status="failed",
            mne_error_type=type(error).__name__,
            mne_error=str(error),
        )

    result["structural_issues"] = "; ".join(dict.fromkeys(structural_issues))
    result["status"] = "needs_replacement" if structural_issues else (
        "optional_metadata_issue" if invalid_optional_patient_fields else "structurally_readable"
    )
    return result


def run_diagnostics(config: dict[str, Any], recording_ids: list[str] | None = None) -> pd.DataFrame:
    # Imported here so the byte-level inspector remains usable without loading
    # PyTorch; path resolution itself is shared with the production pipeline.
    from .preprocess import resolve_edf_path

    paths = config["paths"]
    split_path = resolve_path(config, paths["recording_splits_csv"])
    manifest_path = resolve_path(config, paths["processed_manifest_csv"])
    dataset_root = resolve_path(config, paths.get("dataset_root"))
    assert split_path is not None and manifest_path is not None
    failure_path = manifest_path.with_name("preprocessing_failures.csv")
    output_path = manifest_path.with_name("edf_diagnostics.csv")

    splits = pd.read_csv(split_path)
    if recording_ids:
        selected_ids = set(recording_ids)
    else:
        if not failure_path.exists():
            raise FileNotFoundError(f"No failure manifest found: {failure_path}")
        failure_frame = pd.read_csv(failure_path)
        selected_ids = set(failure_frame["recording_id"].astype(str))
    selected = splits[splits["recording_id"].astype(str).isin(selected_ids)]
    missing = sorted(selected_ids.difference(selected["recording_id"].astype(str)))
    if missing:
        raise ValueError(f"Recording IDs are absent from recording_splits.csv: {missing}")

    rows: list[dict[str, Any]] = []
    for _, split_row in selected.iterrows():
        recording_id = str(split_row["recording_id"])
        try:
            edf_path = resolve_edf_path(split_row, dataset_root)
            diagnostic = inspect_edf(edf_path)
        except Exception as error:
            diagnostic = {
                "status": "path_or_diagnostic_failure",
                "mne_error_type": type(error).__name__,
                "mne_error": str(error),
            }
        diagnostic.update(
            recording_id=recording_id,
            official_split=split_row.get("official_split"),
            label=split_row.get("label"),
        )
        rows.append(diagnostic)
        print(f"\n{recording_id}: {diagnostic['status']}")
        for key in (
            "edf_path",
            "file_size_bytes",
            "invalid_optional_patient_fields",
            "declared_data_records",
            "complete_data_records",
            "mne_n_times",
            "mne_error",
            "structural_issues",
        ):
            if diagnostic.get(key) not in (None, ""):
                print(f"  {key}: {diagnostic[key]}")

    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)
    print(f"\nSaved diagnostics: {output_path}")
    print("This command did not modify any EDF or processed tensor.")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--recording-id",
        action="append",
        dest="recording_ids",
        help="Inspect one ID; repeat the option to inspect several. Defaults to preprocessing failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, ["paths", "preprocessing"])
    run_diagnostics(config, args.recording_ids)


if __name__ == "__main__":
    main()
