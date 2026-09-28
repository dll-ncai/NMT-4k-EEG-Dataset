import numpy as np

from multibknet_nmt.metrics import aggregate_recording_predictions, binary_metrics


def test_recording_probability_is_mean_of_windows():
    result = aggregate_recording_predictions(
        ["a", "a", "b", "b"], [0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9]
    )
    assert np.isclose(
        result.loc[result.recording_id == "a", "abnormal_probability"].item(), 0.2
    )
    assert np.isclose(
        result.loc[result.recording_id == "b", "abnormal_probability"].item(), 0.8
    )


def test_binary_metrics_and_probability_auroc():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1_abnormal"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["tn"] == 2 and metrics["tp"] == 2
