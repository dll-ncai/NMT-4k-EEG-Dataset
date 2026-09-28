import torch
from torch import nn
from torch.utils.data import DataLoader

import multibknet_nmt.train as training


class _CapturedProgress:
    def __init__(self, iterable, postfixes):
        self.iterable = iterable
        self.postfixes = postfixes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, values, refresh=False):
        self.postfixes.append(dict(values))


def _validation_run(monkeypatch, *, best_score, best_epoch, previous_score, epoch):
    postfixes = []

    def fake_tqdm(iterable, **_kwargs):
        return _CapturedProgress(iterable, postfixes)

    monkeypatch.setattr(training, "tqdm", fake_tqdm)
    samples = [
        (torch.tensor([2.0, 0.0]), torch.tensor(0), "normal_edf"),
        (torch.tensor([0.0, 2.0]), torch.tensor(1), "abnormal_edf"),
    ]
    loader = DataLoader(samples, batch_size=2, shuffle=False)
    metrics, _predictions = training._evaluate_dataset(
        nn.Identity(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        amp_enabled=False,
        progress_description="test validation",
        best_score_so_far=best_score,
        best_epoch_so_far=best_epoch,
        previous_epoch_score=previous_score,
        current_epoch=epoch,
    )
    return metrics, postfixes[-1]


def test_validation_progress_marks_new_best(monkeypatch):
    metrics, final_postfix = _validation_run(
        monkeypatch,
        best_score=0.80,
        best_epoch=2,
        previous_score=0.75,
        epoch=3,
    )
    assert metrics["recording"]["accuracy"] == 1.0
    assert metrics["recording"]["auroc"] == 1.0
    assert final_postfix["rec_acc"] == "1.0000"
    assert final_postfix["rec_AUROC"] == "1.0000"
    assert final_postfix["delta"] == "+0.2500"
    assert final_postfix["status"] == "NEW BEST"
    assert final_postfix["best"] == "E03/1.0000"


def test_validation_progress_does_not_replace_tied_best(monkeypatch):
    _metrics, final_postfix = _validation_run(
        monkeypatch,
        best_score=1.0,
        best_epoch=2,
        previous_score=1.0,
        epoch=3,
    )
    assert final_postfix["delta"] == "+0.0000"
    assert final_postfix["status"] == "NO IMPROVEMENT"
    assert final_postfix["best"] == "E02/1.0000"
