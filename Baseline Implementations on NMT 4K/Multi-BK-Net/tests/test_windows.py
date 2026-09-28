from multibknet_nmt.data import compute_window_starts


def test_exact_non_overlapping_windows():
    assert compute_window_starts(12000, 6000, 6000, True) == [0, 6000]


def test_final_window_aligns_to_end():
    assert compute_window_starts(13000, 6000, 6000, True) == [0, 6000, 7000]


def test_final_remainder_can_be_dropped():
    assert compute_window_starts(13000, 6000, 6000, False) == [0, 6000]


def test_too_short_recording_has_no_window():
    assert compute_window_starts(5999, 6000, 6000, True) == []
