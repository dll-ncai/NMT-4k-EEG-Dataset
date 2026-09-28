"""Tests for training-only allowance of missing evaluation EDFs."""

import unittest

import pandas as pd

from eegpt_nmt.cohort import validate_manifest_for_training


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recording_id": "train_0",
                "patient_id": "patient_0",
                "official_split": "train",
                "experiment_split": "internal_train",
                "label": 0,
                "tensor_path": "train_0.pt",
            },
            {
                "recording_id": "train_1",
                "patient_id": "patient_1",
                "official_split": "train",
                "experiment_split": "internal_train",
                "label": 1,
                "tensor_path": "train_1.pt",
            },
            {
                "recording_id": "train_2",
                "patient_id": "patient_2",
                "official_split": "train",
                "experiment_split": "validation",
                "label": 0,
                "tensor_path": "train_2.pt",
            },
            {
                "recording_id": "train_3",
                "patient_id": "patient_3",
                "official_split": "train",
                "experiment_split": "validation",
                "label": 1,
                "tensor_path": "train_3.pt",
            },
            {
                "recording_id": "evaluation_0",
                "patient_id": "evaluation_patient_0",
                "official_split": "evaluation",
                "experiment_split": "evaluation",
                "label": 0,
                "tensor_path": "evaluation_0.pt",
            },
        ]
    )


def _config(allow: bool) -> dict:
    return {
        "split": {
            "expected_train_recordings": 4,
            "expected_evaluation_recordings": 2,
        },
        "training": {"allow_incomplete_evaluation_for_training": allow},
    }


class CohortTests(unittest.TestCase):
    def test_complete_training_allows_deferred_evaluation(self) -> None:
        summary = validate_manifest_for_training(_manifest(), _config(True))
        self.assertEqual(summary["actual_train"], 4)
        self.assertEqual(summary["missing_evaluation"], 1)

    def test_deferred_evaluation_requires_explicit_flag(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Official evaluation is incomplete"):
            validate_manifest_for_training(_manifest(), _config(False))

    def test_missing_training_recording_always_blocks(self) -> None:
        incomplete = _manifest().iloc[1:].reset_index(drop=True)
        with self.assertRaisesRegex(RuntimeError, "training cohort is incomplete"):
            validate_manifest_for_training(incomplete, _config(True))


if __name__ == "__main__":
    unittest.main()
