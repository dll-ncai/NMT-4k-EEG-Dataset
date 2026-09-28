import pytest

from multibknet_nmt.cropping import (
    SHORT_FALLBACK_STRATEGY,
    STANDARD_CROP_STRATEGY,
    select_crop_bounds,
)


def bounds(duration_seconds, policy="last_complete_window"):
    return select_crop_bounds(
        n_times=round(duration_seconds * 200),
        sfreq=200,
        crop_start_seconds=60,
        max_duration_seconds=1200,
        window_seconds=60,
        short_recording_policy=policy,
    )


def test_standard_recording_keeps_paper_crop():
    assert bounds(600) == (12000, 120000, STANDARD_CROP_STRATEGY)


def test_short_recording_uses_last_complete_window():
    assert bounds(90) == (6000, 18000, SHORT_FALLBACK_STRATEGY)


def test_exactly_one_window_uses_real_samples_without_padding():
    assert bounds(60) == (0, 12000, SHORT_FALLBACK_STRATEGY)


def test_recording_shorter_than_window_is_rejected():
    with pytest.raises(ValueError, match="shorter than one complete"):
        bounds(59.9)


def test_strict_policy_preserves_original_rejection():
    with pytest.raises(ValueError, match="less than one complete"):
        bounds(90, policy="strict")
