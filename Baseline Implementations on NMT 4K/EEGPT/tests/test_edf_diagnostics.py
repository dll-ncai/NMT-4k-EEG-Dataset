"""Byte-level regression tests for the failed-EDF diagnostics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eegpt_nmt.diagnose_edfs import inspect_edf


def _pad(value: str, width: int) -> bytes:
    encoded = value.encode("latin-1")
    if len(encoded) > width:
        raise ValueError(f"{value!r} exceeds {width} EDF bytes")
    return encoded.ljust(width, b" ")


def _make_edf(path: Path, include_signal_data: bool) -> None:
    n_signals = 2
    samples_per_record = 200
    header_nbytes = 256 + 256 * n_signals
    fixed = b"".join(
        [
            _pad("0", 8),
            _pad("X X X X height=", 80),
            _pad("Startdate X X X X", 80),
            _pad("01.01.24", 8),
            _pad("00.00.00", 8),
            _pad(str(header_nbytes), 8),
            _pad("", 44),
            _pad("1", 8),
            _pad("1", 8),
            _pad(str(n_signals), 4),
        ]
    )

    def signal_field(values: list[str], width: int) -> bytes:
        return b"".join(_pad(value, width) for value in values)

    signal_header = b"".join(
        [
            signal_field(["FP1", "FP2"], 16),
            signal_field(["", ""], 80),
            signal_field(["uV", "uV"], 8),
            signal_field(["-100", "-100"], 8),
            signal_field(["100", "100"], 8),
            signal_field(["-32768", "-32768"], 8),
            signal_field(["32767", "32767"], 8),
            signal_field(["", ""], 80),
            signal_field([str(samples_per_record)] * n_signals, 8),
            signal_field(["", ""], 32),
        ]
    )
    payload = np.zeros(n_signals * samples_per_record, dtype="<i2").tobytes()
    path.write_bytes(fixed + signal_header + (payload if include_signal_data else b""))


class EdfDiagnosticTests(unittest.TestCase):
    def test_blank_optional_patient_value_is_not_signal_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optional_metadata.edf"
            _make_edf(path, include_signal_data=True)
            result = inspect_edf(path)
        self.assertEqual(result["status"], "optional_metadata_issue")
        self.assertEqual(result["invalid_optional_patient_fields"], "height=''")
        self.assertEqual(result["complete_data_records"], 1)
        self.assertEqual(result["mne_n_times"], 200)

    def test_header_only_edf_requires_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header_only.edf"
            _make_edf(path, include_signal_data=False)
            result = inspect_edf(path)
        self.assertEqual(result["status"], "needs_replacement")
        self.assertEqual(result["complete_data_records"], 0)
        self.assertIn("no waveform data bytes", result["structural_issues"])


if __name__ == "__main__":
    unittest.main()
