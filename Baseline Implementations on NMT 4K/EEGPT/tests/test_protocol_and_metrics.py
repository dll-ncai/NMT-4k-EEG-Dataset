"""Fast tests that do not require EDF data or an EEGPT checkpoint."""

import unittest

import numpy as np
import pandas as pd

from eegpt_nmt.metrics import binary_metrics, select_threshold
from eegpt_nmt.protocol import canonicalize_metadata, create_experiment_splits


class ProtocolTests(unittest.TestCase):
    def test_evaluation_is_never_used_as_validation(self) -> None:
        rows = []
        for index in range(100):
            rows.append(
                {
                    "recording_id": f"train_{index}",
                    "patient_id": f"patient_{index}",
                    "label": index % 2,
                    "split": "train",
                    "Hospital": "a" if index % 3 else "b",
                }
            )
        for index in range(20):
            rows.append(
                {
                    "recording_id": f"eval_{index}",
                    "patient_id": f"eval_patient_{index}",
                    "label": index % 2,
                    "split": "evaluation",
                    "Hospital": "a",
                }
            )
        canonical = canonicalize_metadata(pd.DataFrame(rows))
        result = create_experiment_splits(canonical, 5, 0, 2026, True)
        evaluation = result[result["official_split"] == "evaluation"]
        self.assertTrue(evaluation["experiment_split"].eq("evaluation").all())
        train_patients = set(result.loc[result["experiment_split"] == "internal_train", "patient_id"])
        val_patients = set(result.loc[result["experiment_split"] == "validation", "patient_id"])
        self.assertFalse(train_patients.intersection(val_patients))


class MetricTests(unittest.TestCase):
    def test_metrics_use_recording_predictions(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        result = binary_metrics(labels, probabilities, 0.5)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["tn"], 2)
        self.assertEqual(result["tp"], 2)

    def test_threshold_is_deterministic(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.45, 0.55, 0.9])
        first, _ = select_threshold(labels, probabilities)
        second, _ = select_threshold(labels, probabilities)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

