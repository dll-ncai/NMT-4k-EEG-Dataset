from __future__ import annotations

STANDARD_CROP_STRATEGY = "paper_standard_60s_crop"
SHORT_FALLBACK_STRATEGY = "short_recording_last_complete_window"


def select_crop_bounds(
    *,
    n_times: int,
    sfreq: float,
    crop_start_seconds: float,
    max_duration_seconds: float,
    window_seconds: float,
    short_recording_policy: str,
) -> tuple[int, int, str]:
    """Return an exclusive sample interval and an auditable crop strategy."""
    n_times = int(n_times)
    sfreq = float(sfreq)
    crop_start_seconds = float(crop_start_seconds)
    max_duration_seconds = float(max_duration_seconds)
    window_seconds = float(window_seconds)
    policy = str(short_recording_policy).strip().lower()

    if policy not in {"strict", "last_complete_window"}:
        raise ValueError(
            "preprocessing.short_recording_policy must be strict or "
            "last_complete_window"
        )
    if n_times <= 0 or sfreq <= 0:
        raise ValueError("recording sample count and sampling rate must be positive")
    if crop_start_seconds < 0 or max_duration_seconds <= 0 or window_seconds <= 0:
        raise ValueError("crop/window durations must be positive")

    requested_start = round(crop_start_seconds * sfreq)
    maximum_samples = round(max_duration_seconds * sfreq)
    window_samples = round(window_seconds * sfreq)
    requested_stop = min(n_times, requested_start + maximum_samples)

    if (
        requested_start < n_times
        and requested_stop - requested_start >= window_samples
    ):
        return requested_start, requested_stop, STANDARD_CROP_STRATEGY

    # NMT-4K contains a few recordings too short to discard the first minute
    # and still retain a complete 60-second model input. Use the latest complete
    # window when possible. This maximizes the discarded lead-in without
    # padding, repeating, or fabricating EEG samples.
    if policy == "last_complete_window" and n_times >= window_samples:
        return (
            n_times - window_samples,
            n_times,
            SHORT_FALLBACK_STRATEGY,
        )

    if n_times < window_samples:
        raise ValueError(
            f"recording is shorter than one complete {window_seconds:g}-second window"
        )
    if requested_start >= n_times:
        raise ValueError(
            f"recording ends at or before the configured "
            f"{crop_start_seconds:g}-second crop point"
        )
    raise ValueError(
        f"less than one complete {window_seconds:g}-second window remains after cropping"
    )
