import numpy as np
import pandas as pd
import pytest

import multibknet_nmt.data as data_module
from multibknet_nmt.data import WindowedRecordingDataset
from multibknet_nmt.train import _runtime_integer


def test_runtime_integer_reads_environment_without_changing_config(monkeypatch):
    monkeypatch.setenv("MBK_TEST_RUNTIME_VALUE", "64")
    assert _runtime_integer("MBK_TEST_RUNTIME_VALUE", 16) == 64


def test_runtime_integer_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("MBK_TEST_RUNTIME_VALUE", "0")
    with pytest.raises(ValueError, match="at least 1"):
        _runtime_integer("MBK_TEST_RUNTIME_VALUE", 16)


def test_dataset_reuses_cached_memory_map(monkeypatch, tmp_path):
    cache_path = tmp_path / "recording.npy"
    np.save(cache_path, np.zeros((19, 12000), dtype=np.float32))
    frame = pd.DataFrame(
        [
            {
                "recording_id": "example",
                "label": 0,
                "cache_path": str(cache_path),
                "n_samples": 12000,
            }
        ]
    )
    monkeypatch.setenv("MBK_RUNTIME_MEMMAP_CACHE_SIZE", "8")
    real_load = np.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(data_module.np, "load", counted_load)
    dataset = WindowedRecordingDataset(frame, 6000, 6000, True)
    first, _, _ = dataset[0]
    second, _, _ = dataset[1]

    assert first.shape == (19, 6000)
    assert second.shape == (19, 6000)
    assert len(calls) == 1
