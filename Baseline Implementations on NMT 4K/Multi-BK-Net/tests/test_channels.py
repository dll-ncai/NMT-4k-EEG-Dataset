import pytest

from multibknet_nmt.channels import canonicalize_channel, select_channel_indices


def test_common_channel_aliases():
    assert canonicalize_channel("EEG FP1-REF") == "Fp1"
    assert canonicalize_channel("EEG T7-LE") == "T3"
    assert canonicalize_channel("P8") == "T6"
    assert canonicalize_channel("Fp1-F7") is None


def test_ordered_channel_selection():
    raw = ["EEG O2-LE", "EEG FP1-LE", "EEG C3-LE"]
    indices, mapping = select_channel_indices(raw, ["Fp1", "C3", "O2"])
    assert indices == [1, 2, 0]
    assert mapping["Fp1"] == "EEG FP1-LE"


def test_missing_channel_is_not_silently_filled():
    with pytest.raises(ValueError, match="missing channels"):
        select_channel_indices(["EEG FP1-REF"], ["Fp1", "Fp2"])
